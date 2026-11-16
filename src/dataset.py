from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["Negative", "Positive"]  # label 0 = no crack, 1 = crack
MEAN = [0.485, 0.456, 0.406]            # ImageNet statistics
STD = [0.229, 0.224, 0.225]


def list_samples(data_dir):
    samples = []
    for label, cls in enumerate(CLASS_NAMES):
        for p in sorted((Path(data_dir) / cls).glob("*.jpg")):
            samples.append((str(p), label))
    return samples


def split_samples(samples, val_size, test_size, seed):
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]
    p_train, p_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=val_size + test_size,
        stratify=labels, random_state=seed)
    p_val, p_test, y_val, y_test = train_test_split(
        p_tmp, y_tmp, test_size=test_size / (val_size + test_size),
        stratify=y_tmp, random_state=seed)
    return (list(zip(p_train, y_train)),
            list(zip(p_val, y_val)),
            list(zip(p_test, y_test)))


def get_transforms(img_size, train):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class CrackDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label