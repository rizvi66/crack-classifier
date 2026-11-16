import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import (CLASS_NAMES, CrackDataset, get_transforms,
                     list_samples, split_samples)
from model import build_model, set_backbone_trainable


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type=device.type,
                                enabled=device.type == "cuda"):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        out = model(x.to(device))
        ps.extend(out.argmax(1).cpu().tolist())
        ys.extend(y.tolist())
    return ys, ps


def plot_history(history, path):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["train_loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["train_acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()


def main(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    samples = list_samples(cfg["data_dir"])
    assert len(samples) > 0, "No images found - check data_dir in the config"
    train_s, val_s, test_s = split_samples(
        samples, cfg["val_size"], cfg["test_size"], cfg["seed"])
    print(f"train {len(train_s)} | val {len(val_s)} | test {len(test_s)}")

    def loader(s, train):
        ds = CrackDataset(s, get_transforms(cfg["img_size"], train))
        return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=train,
                          num_workers=cfg["num_workers"], pin_memory=True)

    train_dl, val_dl, test_dl = loader(train_s, True), loader(val_s, False), loader(test_s, False)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    out_dir = Path(cfg["output_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    res_dir = Path(cfg["results_dir"]); res_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    optimizer = None

    for epoch in range(cfg["epochs"]):
        if epoch == 0:  # phase 1: train only the new head
            set_backbone_trainable(model, False)
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=cfg["lr_head"])
        elif epoch == cfg["freeze_epochs"]:  # phase 2: fine-tune everything
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr_finetune"])

        tr_loss, tr_acc = run_epoch(model, train_dl, criterion, device, optimizer, scaler)
        va_loss, va_acc = run_epoch(model, val_dl, criterion, device)
        for k, v in zip(history, [tr_loss, va_loss, tr_acc, va_acc]):
            history[k].append(v)
        print(f"epoch {epoch+1}/{cfg['epochs']} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

    # final evaluation on the untouched test set, using the best epoch
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    ys, ps = predict(model, test_dl, device)
    report = classification_report(ys, ps, target_names=CLASS_NAMES, output_dict=True)
    print(classification_report(ys, ps, target_names=CLASS_NAMES, digits=4))

    plot_history(history, res_dir / "training_curves.png")
    plot_confusion(confusion_matrix(ys, ps), res_dir / "confusion_matrix.png")
    json.dump(report, open(res_dir / "test_metrics.json", "w"), indent=2)
    print("Saved results to", res_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    main(parser.parse_args().config)