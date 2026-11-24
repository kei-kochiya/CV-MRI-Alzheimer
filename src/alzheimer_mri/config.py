"""Central configuration and class definitions for Alzheimer's MRI classification."""

from dataclasses import dataclass
from pathlib import Path

CLASS_NAMES: tuple[str, ...] = (
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented",
)

CLASS_LABELS: dict[str, str] = {
    "MildDemented": "Mild Dementia",
    "ModerateDemented": "Moderate Dementia",
    "NonDemented": "Non-Demented",
    "VeryMildDemented": "Very Mild Dementia",
}

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
VALID_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(slots=True)
class TrainConfig:
    """Hyperparameters and paths for two-stage ResNet34 training."""

    data_dir: Path = Path("dataset")
    output_dir: Path = Path("artifacts")
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    warmup_epochs: int = 5
    finetune_epochs: int = 15
    warmup_lr: float = 1e-3
    finetune_lr: float = 1e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    dropout: float = 0.3
    early_stopping_patience: int = 4
    use_amp: bool = True
    use_class_weights: bool = True
    pretrained: bool = True
    device: str = "auto"
    max_batches: int | None = None

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "manifest.parquet"

    @property
    def checkpoint_path(self) -> Path:
        return self.output_dir / "resnet34_best.pt"

    @property
    def history_path(self) -> Path:
        return self.output_dir / "training_history.parquet"
