"""ResNet34 model construction, backbone freezing, and checkpoint utilities."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet34_Weights, resnet34

from alzheimer_mri.config import CLASS_NAMES, TrainConfig


def build_resnet34(
    num_classes: int = len(CLASS_NAMES),
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a ResNet34 classifier with a regularized linear head."""
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    model = resnet34(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze all layers except the final classification head."""
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = trainable


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (total_parameters, trainable_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def resolve_device(requested: str = "auto") -> torch.device:
    """Select execution device ('auto', 'cuda', or 'cpu')."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: TrainConfig,
    epoch: int,
    val_acc: float,
    val_loss: float,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> None:
    """Save ResNet34 state dict and inference metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_config = {
        k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()
    }
    torch.save(
        {
            "arch": "resnet34",
            "state_dict": model.state_dict(),
            "class_names": list(class_names),
            "img_size": config.img_size,
            "dropout": config.dropout,
            "epoch": epoch,
            "val_acc": float(val_acc),
            "val_loss": float(val_loss),
            "config": serialized_config,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a trained ResNet34 model and metadata from a checkpoint."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    dropout = float(checkpoint.get("dropout", 0.3))

    model = build_resnet34(
        num_classes=len(class_names),
        pretrained=False,
        dropout=dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
