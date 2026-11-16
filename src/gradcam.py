import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from dataset import (CLASS_NAMES, CrackDataset, get_transforms,
                     list_samples, split_samples)
from model import build_model


class GradCAM:
    """Grad-CAM for a convolutional layer of a PyTorch model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        output.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None:
            class_idx = out.argmax(1).item()
        out[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear",
                            align_corners=False)[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        probs = out.softmax(1)[0].detach().cpu().numpy()
        return cam.detach().cpu().numpy(), class_idx, probs


def plot_cams(items, cam_engine, tf, img_size, device, path, title):
    cols = 4
    rows = int(np.ceil(len(items) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (img_path, label) in zip(axes, items):
        img = Image.open(img_path).convert("RGB")
        x = tf(img).unsqueeze(0).to(device)
        heat, pred, probs = cam_engine(x)
        base = np.asarray(img.resize((img_size, img_size))) / 255.0
        ax.imshow(base)
        ax.imshow(heat, cmap="jet", alpha=0.4)
        ax.set_title(f"true: {CLASS_NAMES[label]}\n"
                     f"pred: {CLASS_NAMES[pred]} ({probs[pred]:.3f})", fontsize=9)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


def main(cfg_path, weights, n_examples):
    cfg = yaml.safe_load(open(cfg_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(pretrained=False).to(device)
    model.load_state_dict(torch.load(weights, map_location=device))
    model.eval()
    cam_engine = GradCAM(model, model.layer4)

    # Rebuild the exact same test split (same seed as training)
    samples = list_samples(cfg["data_dir"])
    _, _, test_s = split_samples(samples, cfg["val_size"],
                                 cfg["test_size"], cfg["seed"])
    tf = get_transforms(cfg["img_size"], train=False)

    # Predict the whole test set to find mistakes
    dl = DataLoader(CrackDataset(test_s, tf), batch_size=128, shuffle=False,
                    num_workers=cfg["num_workers"])
    preds = []
    with torch.no_grad():
        for x, _ in dl:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())

    errors = [s for s, p in zip(test_s, preds) if p != s[1]]
    print(f"{len(errors)} misclassified images in test set")

    rng = random.Random(cfg["seed"])
    ok_crack = [s for s, p in zip(test_s, preds) if s[1] == 1 and p == 1]
    ok_clean = [s for s, p in zip(test_s, preds) if s[1] == 0 and p == 0]
    examples = rng.sample(ok_crack, n_examples) + rng.sample(ok_clean, n_examples)

    res = Path(cfg["results_dir"])
    res.mkdir(parents=True, exist_ok=True)
    if errors:
        plot_cams(errors, cam_engine, tf, cfg["img_size"], device,
                  res / "gradcam_errors.png", "Grad-CAM: misclassified test images")
    plot_cams(examples, cam_engine, tf, cfg["img_size"], device,
              res / "gradcam_examples.png",
              "Grad-CAM: correct predictions (first rows crack, then no crack)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--n", type=int, default=4, help="examples per class")
    args = parser.parse_args()
    main(args.config, args.weights, args.n)