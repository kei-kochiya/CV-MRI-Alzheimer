"""Polars-backed dataset indexing, stratified splitting, and PyTorch DataLoaders."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from alzheimer_mri.config import (
    CLASS_LABELS,
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_EXTENSIONS,
)


def scan_dataset(
    data_dir: Path,
    class_names: Iterable[str] = CLASS_NAMES,
) -> pl.DataFrame:
    """Scan image folders into a deterministic Polars DataFrame."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    records: list[dict[str, object]] = []
    for label_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            continue

        display_name = CLASS_LABELS.get(class_name, class_name)
        with os.scandir(class_dir) as entries:
            for entry in entries:
                if not entry.is_file():
                    continue
                ext = os.path.splitext(entry.name)[1].lower()
                if ext not in VALID_EXTENSIONS:
                    continue
                records.append(
                    {
                        "rel_path": f"{class_name}/{entry.name}",
                        "abs_path": entry.path,
                        "class_name": class_name,
                        "display_name": display_name,
                        "label_idx": label_idx,
                        "file_bytes": entry.stat().st_size,
                    }
                )

    if not records:
        raise ValueError(f"No valid MRI images found under {data_dir}")

    return pl.DataFrame(records).sort("rel_path")


def assign_stratified_splits(
    df: pl.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pl.DataFrame:
    """Assign deterministic stratified train/val/test splits per class using Polars."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    # Sort before sampling so the split is identical regardless of filesystem scan order.
    shuffled = df.sort("rel_path").sample(fraction=1.0, shuffle=True, seed=seed)

    return (
        shuffled.with_columns(
            [
                pl.int_range(pl.len()).over("class_name").alias("_cls_rank"),
                pl.len().over("class_name").alias("_cls_total"),
            ]
        )
        .with_columns(
            pl.when(pl.col("_cls_rank") < (pl.col("_cls_total") * train_ratio).cast(pl.Int64))
            .then(pl.lit("train"))
            .when(
                pl.col("_cls_rank")
                < (pl.col("_cls_total") * (train_ratio + val_ratio)).cast(pl.Int64)
            )
            .then(pl.lit("val"))
            .otherwise(pl.lit("test"))
            .alias("split")
        )
        .drop(["_cls_rank", "_cls_total"])
        .sort("rel_path")
    )


def build_or_load_manifest(
    data_dir: Path,
    manifest_path: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    force_rebuild: bool = False,
) -> pl.DataFrame:
    """Build or load the dataset Parquet manifest."""
    manifest_path = Path(manifest_path)
    if manifest_path.is_file() and not force_rebuild:
        return pl.read_parquet(manifest_path)

    raw_df = scan_dataset(data_dir)
    manifest_df = assign_stratified_splits(
        raw_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.write_parquet(manifest_path)
    return manifest_df


def summarize_manifest(df: pl.DataFrame) -> pl.DataFrame:
    """Return per-split, per-class counts and mean file sizes via Polars."""
    return (
        df.group_by(["split", "label_idx", "class_name"])
        .agg(
            [
                pl.len().alias("count"),
                (pl.col("file_bytes").mean() / 1024.0).round(2).alias("avg_kb"),
            ]
        )
        .sort(["split", "label_idx"])
    )


def compute_class_weights(df: pl.DataFrame, num_classes: int = len(CLASS_NAMES)) -> list[float]:
    """Compute balanced inverse-frequency class weights from the training split."""
    train_counts = (
        df.filter(pl.col("split") == "train")
        .group_by("label_idx")
        .agg(pl.len().alias("count"))
        .sort("label_idx")
    )
    counts_map = dict(zip(train_counts["label_idx"].to_list(), train_counts["count"].to_list()))
    total = sum(counts_map.values())
    if total == 0:
        return [1.0] * num_classes

    weights = []
    for idx in range(num_classes):
        count = counts_map.get(idx, 1)
        weights.append(total / (num_classes * count))
    return weights


def get_transforms(img_size: int = 224, is_train: bool = False) -> transforms.Compose:
    """Return MRI-appropriate torchvision transforms for ResNet34."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class AlzheimerMRIDataset(Dataset[tuple[torch.Tensor, int]]):
    """PyTorch Dataset backed by a Polars manifest DataFrame."""

    def __init__(self, manifest_df: pl.DataFrame, transform: transforms.Compose | None = None):
        self.paths = manifest_df["abs_path"].to_list()
        self.labels = manifest_df["label_idx"].to_list()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as img:
            rgb_img = img.convert("RGB")
            tensor = self.transform(rgb_img) if self.transform else transforms.ToTensor()(rgb_img)
        return tensor, int(self.labels[index])


def create_dataloaders(
    manifest_df: pl.DataFrame,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = False,
) -> dict[str, DataLoader[tuple[torch.Tensor, int]]]:
    """Create train, validation, and test PyTorch DataLoaders from a Polars manifest."""
    loaders: dict[str, DataLoader[tuple[torch.Tensor, int]]] = {}
    persistent = num_workers > 0

    for split_name in ("train", "val", "test"):
        split_df = manifest_df.filter(pl.col("split") == split_name)
        is_train = split_name == "train"
        dataset = AlzheimerMRIDataset(
            split_df,
            transform=get_transforms(img_size=img_size, is_train=is_train),
        )
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent,
        )
    return loaders
