"""Two-stage PyTorch ResNet34 training script with AMP and Polars metric logging."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alzheimer_mri.config import CLASS_NAMES, TrainConfig
from alzheimer_mri.data import (
    build_or_load_manifest,
    compute_class_weights,
    create_dataloaders,
    summarize_manifest,
)
from alzheimer_mri.model import (
    build_resnet34,
    count_parameters,
    resolve_device,
    save_checkpoint,
    set_backbone_trainable,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp_enabled: bool = False,
    max_batches: int | None = None,
    desc: str = "",
) -> tuple[float, float]:
    """Run one training or evaluation epoch and return (avg_loss, accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch_idx, (images, labels) in enumerate(progress):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train and optimizer is not None:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0
    return total_loss / total_samples, total_correct / total_samples


def plot_training_curves(history_df: pl.DataFrame, output_path: Path, warmup_epochs: int) -> None:
    """Save accuracy and loss curves from the Polars training history DataFrame."""
    if history_df.is_empty():
        return

    epochs = list(range(1, history_df.height + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history_df["train_acc"].to_list(), label="Train Accuracy")
    axes[0].plot(epochs, history_df["val_acc"].to_list(), label="Val Accuracy")
    if 0 < warmup_epochs < len(epochs):
        axes[0].axvline(x=warmup_epochs, color="#666666", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history_df["train_loss"].to_list(), label="Train Loss")
    axes[1].plot(epochs, history_df["val_loss"].to_list(), label="Val Loss")
    if 0 < warmup_epochs < len(epochs):
        axes[1].axvline(x=warmup_epochs, color="#666666", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train(config: TrainConfig) -> pl.DataFrame:
    """Execute two-stage ResNet34 training and return the Polars history DataFrame."""
    set_seed(config.seed)
    device = resolve_device(config.device)
    amp_enabled = config.use_amp and device.type == "cuda"

    manifest_df = build_or_load_manifest(
        data_dir=config.data_dir,
        manifest_path=config.manifest_path,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.seed,
    )
    print("Dataset split summary:")
    print(summarize_manifest(manifest_df))

    loaders = create_dataloaders(
        manifest_df=manifest_df,
        img_size=config.img_size,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_resnet34(
        num_classes=len(CLASS_NAMES),
        pretrained=config.pretrained,
        dropout=config.dropout,
    ).to(device)

    weight_tensor = None
    if config.use_class_weights:
        weights = compute_class_weights(manifest_df, num_classes=len(CLASS_NAMES))
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=config.label_smoothing,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history_rows: list[dict[str, object]] = []
    best_val_acc = -1.0
    best_val_loss = float("inf")
    global_epoch = 0

    stages = [
        ("warmup", config.warmup_epochs, config.warmup_lr, False),
        ("finetune", config.finetune_epochs, config.finetune_lr, True),
    ]

    for stage_name, num_epochs, lr, backbone_trainable in stages:
        if num_epochs <= 0:
            continue

        set_backbone_trainable(model, trainable=backbone_trainable)
        total_p, trainable_p = count_parameters(model)
        print(
            f"\n[{stage_name.upper()}] device={device} amp={amp_enabled} "
            f"params={trainable_p:,}/{total_p:,} lr={lr}"
        )

        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=config.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))
        epochs_without_improvement = 0

        for stage_epoch in range(1, num_epochs + 1):
            global_epoch += 1
            t0 = time.perf_counter()

            train_loss, train_acc = run_epoch(
                model=model,
                loader=loaders["train"],
                criterion=criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                max_batches=config.max_batches,
                desc=f"{stage_name} {stage_epoch}/{num_epochs} [train]",
            )
            val_loss, val_acc = run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                amp_enabled=amp_enabled,
                max_batches=config.max_batches,
                desc=f"{stage_name} {stage_epoch}/{num_epochs} [val]",
            )
            scheduler.step()
            elapsed = time.perf_counter() - t0
            current_lr = float(optimizer.param_groups[0]["lr"])

            history_rows.append(
                {
                    "global_epoch": global_epoch,
                    "stage": stage_name,
                    "stage_epoch": stage_epoch,
                    "train_loss": round(train_loss, 5),
                    "train_acc": round(train_acc, 5),
                    "val_loss": round(val_loss, 5),
                    "val_acc": round(val_acc, 5),
                    "lr": current_lr,
                    "duration_sec": round(elapsed, 2),
                }
            )
            print(
                f"Epoch {global_epoch:02d} ({stage_name}) | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
                f"time={elapsed:.1f}s"
            )

            improved = val_acc > best_val_acc or (
                abs(val_acc - best_val_acc) < 1e-5 and val_loss < best_val_loss
            )
            if improved:
                best_val_acc = val_acc
                best_val_loss = val_loss
                epochs_without_improvement = 0
                save_checkpoint(
                    path=config.checkpoint_path,
                    model=model,
                    config=config,
                    epoch=global_epoch,
                    val_acc=val_acc,
                    val_loss=val_loss,
                )
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.early_stopping_patience:
                    print(f"Early stopping triggered in {stage_name} at epoch {global_epoch}.")
                    break

    history_df = pl.DataFrame(history_rows)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history_df.write_parquet(config.history_path)
    history_df.write_csv(config.output_dir / "training_history.csv")
    plot_training_curves(
        history_df,
        output_path=config.output_dir / "training_curves.png",
        warmup_epochs=config.warmup_epochs,
    )
    print(f"\nBest validation accuracy: {best_val_acc:.4f} -> {config.checkpoint_path}")
    return history_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet34 on Alzheimer's MRI dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--warmup-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Run 1 warmup + 1 finetune epoch with 2 batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=0 if args.dry_run else args.num_workers,
        warmup_epochs=1 if args.dry_run else args.warmup_epochs,
        finetune_epochs=1 if args.dry_run else args.finetune_epochs,
        warmup_lr=args.warmup_lr,
        finetune_lr=args.finetune_lr,
        device=args.device,
        use_amp=not args.no_amp,
        pretrained=not args.no_pretrained,
        max_batches=2 if args.dry_run else None,
    )
    train(cfg)


if __name__ == "__main__":
    main()
