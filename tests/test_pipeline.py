"""CPU unit and smoke tests for the PyTorch ResNet34 + Polars pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from PIL import Image

from alzheimer_mri.config import CLASS_NAMES, TrainConfig
from alzheimer_mri.data import (
    assign_stratified_splits,
    build_or_load_manifest,
    compute_class_weights,
    scan_dataset,
    summarize_manifest,
)
from alzheimer_mri.gradcam import GradCAM, render_overlay
from alzheimer_mri.model import (
    build_resnet34,
    count_parameters,
    load_checkpoint,
    save_checkpoint,
    set_backbone_trainable,
)
from alzheimer_mri.predict import AlzheimerPredictor, create_pdf_report
from alzheimer_mri.train import train


def _create_synthetic_dataset(root: Path, per_class: int = 10) -> Path:
    rng = np.random.default_rng(42)
    for cls in CLASS_NAMES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(per_class):
            arr = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
            Image.fromarray(arr, mode="RGB").save(cls_dir / f"sample_{idx:03d}.jpg")
    return root


def test_polars_manifest_and_stratified_split(tmp_path: Path) -> None:
    data_dir = _create_synthetic_dataset(tmp_path / "dataset", per_class=20)
    raw = scan_dataset(data_dir)
    assert raw.height == 80

    manifest = assign_stratified_splits(raw, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)
    assert manifest.height == 80

    # Zero leakage between splits
    train_paths = set(manifest.filter(pl.col("split") == "train")["rel_path"].to_list())
    val_paths = set(manifest.filter(pl.col("split") == "val")["rel_path"].to_list())
    test_paths = set(manifest.filter(pl.col("split") == "test")["rel_path"].to_list())
    assert train_paths.isdisjoint(val_paths)
    assert train_paths.isdisjoint(test_paths)
    assert val_paths.isdisjoint(test_paths)

    summary = summarize_manifest(manifest)
    assert summary.height == 12  # 3 splits * 4 classes

    weights = compute_class_weights(manifest)
    assert len(weights) == 4
    assert all(pytest.approx(1.0, abs=1e-6) == w for w in weights)


def test_resnet34_build_freeze_and_checkpoint(tmp_path: Path) -> None:
    model = build_resnet34(num_classes=4, pretrained=False, dropout=0.3)
    total_params, trainable_all = count_parameters(model)
    assert total_params == trainable_all
    assert 21_000_000 < total_params < 22_000_000

    set_backbone_trainable(model, trainable=False)
    _, trainable_head = count_parameters(model)
    assert trainable_head == 512 * 4 + 4

    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 4)

    ckpt_path = tmp_path / "resnet34_best.pt"
    cfg = TrainConfig(data_dir=tmp_path, output_dir=tmp_path, pretrained=False, device="cpu")
    save_checkpoint(ckpt_path, model, cfg, epoch=1, val_acc=0.75, val_loss=0.6)

    loaded_model, meta = load_checkpoint(ckpt_path, device="cpu")
    assert meta["arch"] == "resnet34"
    assert meta["epoch"] == 1
    with torch.inference_mode():
        loaded_logits = loaded_model(x)
    assert loaded_logits.shape == (2, 4)


def test_gradcam_and_pdf_report(tmp_path: Path) -> None:
    model = build_resnet34(num_classes=4, pretrained=False)
    x = torch.randn(1, 3, 128, 128)
    with GradCAM(model) as cam:
        heatmap, logits = cam.generate(x)

    assert logits.shape == (1, 4)
    assert heatmap.shape == (128, 128)
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) <= 1.0

    img = Image.new("RGB", (180, 180), color=(110, 110, 110))
    overlay = render_overlay(img, heatmap)
    assert overlay.size == (180, 180)

    predictor = AlzheimerPredictor(
        checkpoint_path=tmp_path / "missing.pt",
        device_name="cpu",
        allow_untrained_fallback=True,
    )
    res = predictor.predict_image(img)
    assert res.model_mode == "preview_untrained"
    pdf_bytes = create_pdf_report(img, res.heatmap_overlay, res)
    assert pdf_bytes.startswith(b"%PDF")


def test_cpu_training_smoke(tmp_path: Path) -> None:
    data_dir = _create_synthetic_dataset(tmp_path / "dataset", per_class=8)
    out_dir = tmp_path / "artifacts"
    cfg = TrainConfig(
        data_dir=data_dir,
        output_dir=out_dir,
        img_size=128,
        batch_size=4,
        num_workers=0,
        warmup_epochs=1,
        finetune_epochs=1,
        use_amp=False,
        pretrained=False,
        device="cpu",
        max_batches=2,
    )
    history = train(cfg)
    assert history.height == 2
    assert cfg.checkpoint_path.is_file()
    assert cfg.history_path.is_file()
    assert (out_dir / "training_curves.png").is_file()


def test_real_dataset_manifest_read(tmp_path: Path) -> None:
    dataset_dir = Path("dataset")
    if not dataset_dir.is_dir():
        pytest.skip("Real dataset not present")
    manifest = build_or_load_manifest(
        data_dir=dataset_dir,
        manifest_path=tmp_path / "manifest.parquet",
        force_rebuild=True,
    )
    assert manifest.height == 44_000


def test_fastapi_endpoints() -> None:
    from fastapi.testclient import TestClient

    import app as web_app

    client = TestClient(web_app.app)

    res_index = client.get("/")
    assert res_index.status_code == 200
    assert "Axial brain MRI dementia staging" in res_index.text

    res_tokens = client.get("/tokens.css")
    assert res_tokens.status_code == 200
    assert "--color-paper" in res_tokens.text

    res_favicon = client.get("/favicon.ico")
    assert res_favicon.status_code == 200

    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    status_json = res_status.json()
    assert status_json["arch"] == "ResNet34"
    assert status_json["total_params"] > 21_000_000

    res_samples = client.get("/api/samples")
    assert res_samples.status_code == 200
    samples = res_samples.json()["items"]
    if samples:
        first_cls = samples[0]["class_name"]
        res_pred = client.post("/api/predict", data={"sample_class": first_cls, "alpha": "0.45"})
        assert res_pred.status_code == 200
        pred_json = res_pred.json()
        assert "probabilities" in pred_json
        assert pred_json["original_data_url"].startswith("data:image/png;base64,")
        assert pred_json["heatmap_only_data_url"].startswith("data:image/png;base64,")

        res_pdf = client.post("/api/report", data={"sample_class": first_cls, "alpha": "0.45"})
        assert res_pdf.status_code == 200
        assert res_pdf.headers["content-type"] == "application/pdf"
        assert res_pdf.content.startswith(b"%PDF")

