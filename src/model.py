import torch.nn as nn
from torchvision import models


def build_model(num_classes=2, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new 2-output head
    return model


def set_backbone_trainable(model, trainable):
    """Freeze or unfreeze everything except the final layer (fc)."""
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = trainable