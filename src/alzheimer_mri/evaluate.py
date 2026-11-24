"""Test-split evaluation with Polars prediction logs and confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alzheimer_mri.config import CLASS_NAMES
from alzheimer_mri.data import (
    AlzheimerMRIDataset,
    build_or_load_manifest,
    get_transforms,
)
from alzheimer_mri.model import load_checkpoint, resolve_device


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: tuple[str, ...],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title="ResNet34 Test Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


@torch.inference_mode()
def evaluate(
    checkpoint_path: Path,
    data_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    batch_size: int = 64,
    num_workers: int = 4,
    device_name: str = "auto",
) -> dict[str, object]:
    """Evaluate a ResNet34 checkpoint on the held-out test split."""
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    img_size = int(checkpoint.get("img_size", 224))
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))

    manifest_df = build_or_load_manifest(data_dir=data_dir, manifest_path=manifest_path)
    test_df = manifest_df.filter(pl.col("split") == "test")

    dataset = AlzheimerMRIDataset(test_df, transform=get_transforms(img_size=img_size, is_train=False))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs: list[np.ndarray] = []
    all_preds: list[int] = []
    all_labels: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1).tolist()

        all_probs.append(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    probs_matrix = np.concatenate(all_probs, axis=0)
    pred_classes = [class_names[idx] for idx in all_preds]
    confidences = probs_matrix.max(axis=1).tolist()

    results_df = test_df.with_columns(
        [
            pl.Series("pred_idx", all_preds),
            pl.Series("pred_class", pred_classes),
            pl.Series("confidence", confidences),
            pl.Series("correct", [p == y for p, y in zip(all_preds, all_labels)]),
            *(
                pl.Series(f"prob_{cls}", probs_matrix[:, i].tolist())
                for i, cls in enumerate(class_names)
            ),
        ]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    results_df.write_parquet(output_dir / "test_predictions.parquet")

    acc = float(accuracy_score(all_labels, all_preds))
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, output_dir / "confusion_matrix.png")

    report = classification_report(all_labels, all_preds, target_names=list(class_names), digits=4)
    print(f"Test Accuracy: {acc * 100:.2f}% | Macro F1: {macro_f1:.4f}")
    print(report)

    per_class_acc = (
        results_df.group_by("class_name")
        .agg(
            [
                pl.len().alias("samples"),
                pl.col("correct").mean().round(4).alias("accuracy"),
                pl.col("confidence").mean().round(4).alias("avg_confidence"),
            ]
        )
        .sort("class_name")
    )
    print(per_class_acc)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "num_samples": len(all_labels),
        "predictions_path": str(output_dir / "test_predictions.parquet"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained ResNet34 on test split.")
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/resnet34_best.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifest.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
