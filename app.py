"""FastAPI web workbench for PyTorch ResNet34 Alzheimer's MRI screening."""

from __future__ import annotations

import argparse
import io
import os
import time
from pathlib import Path
from typing import Annotated

import matplotlib
import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from alzheimer_mri.config import CLASS_LABELS, CLASS_NAMES, VALID_EXTENSIONS
from alzheimer_mri.model import count_parameters
from alzheimer_mri.predict import (
    AlzheimerPredictor,
    create_pdf_report,
    image_to_data_url,
)

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
DATASET_DIR = BASE_DIR / "dataset"
MANIFEST_PATH = BASE_DIR / "artifacts" / "manifest.parquet"
CHECKPOINT_PATH = Path(os.environ.get("MRI_CHECKPOINT", BASE_DIR / "artifacts" / "resnet34_best.pt"))
DEVICE_NAME = os.environ.get("MRI_DEVICE", "cpu")

app = FastAPI(title="neuro-mri · ResNet34 Alzheimer Screening", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_predictor: AlzheimerPredictor | None = None


def get_predictor() -> AlzheimerPredictor:
    global _predictor
    if _predictor is None:
        _predictor = AlzheimerPredictor(
            checkpoint_path=CHECKPOINT_PATH,
            device_name=DEVICE_NAME,
            allow_untrained_fallback=True,
        )
    return _predictor


def _heatmap_only_image(heatmap: np.ndarray, size: tuple[int, int]) -> Image.Image:
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    resized = Image.fromarray(heatmap_uint8, mode="L").resize(size, resample=Image.Resampling.BILINEAR)
    norm = np.asarray(resized, dtype=np.float32) / 255.0
    colored = (matplotlib.colormaps["turbo"](norm)[..., :3] * 255.0).astype(np.uint8)
    return Image.fromarray(colored, mode="RGB")


def _dataset_counts_via_polars() -> dict[str, int]:
    if MANIFEST_PATH.is_file():
        df = pl.read_parquet(MANIFEST_PATH)
        grouped = df.group_by("class_name").agg(pl.len().alias("count"))
        return dict(zip(grouped["class_name"].to_list(), grouped["count"].to_list()))

    counts: dict[str, int] = {}
    if DATASET_DIR.is_dir():
        for cls in CLASS_NAMES:
            cls_dir = DATASET_DIR / cls
            if cls_dir.is_dir():
                counts[cls] = sum(
                    1 for entry in os.scandir(cls_dir) if entry.is_file()
                )
    return counts


def _pick_sample_file(class_name: str) -> Path:
    if class_name not in CLASS_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown class preset: {class_name}")
    cls_dir = DATASET_DIR / class_name
    if not cls_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Sample directory not found for {class_name}")

    for entry in sorted(os.scandir(cls_dir), key=lambda e: e.name):
        if entry.is_file() and Path(entry.name).suffix.lower() in VALID_EXTENSIONS:
            return Path(entry.path)
    raise HTTPException(status_code=404, detail=f"No sample MRI images found in {class_name}")


async def _load_input_image(file: UploadFile | None, sample_class: str | None) -> Image.Image:
    if file is not None and file.filename:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            with Image.open(io.BytesIO(raw_bytes)) as img:
                return img.convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    if sample_class:
        sample_path = _pick_sample_file(sample_class)
        with Image.open(sample_path) as img:
            return img.convert("RGB")

    raise HTTPException(status_code=400, detail="Provide either an uploaded MRI file or a sample preset.")


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/tokens.css")
def serve_tokens() -> FileResponse:
    return FileResponse(BASE_DIR / "tokens.css", media_type="text/css")


@app.get("/favicon.ico")
def serve_favicon() -> FileResponse:
    return FileResponse(WEB_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/api/status")
def api_status() -> dict[str, object]:
    predictor = get_predictor()
    total_params, _ = count_parameters(predictor.model)
    return {
        "arch": "ResNet34",
        "model_mode": predictor.model_mode,
        "val_acc": predictor.val_acc,
        "epoch": predictor.epoch,
        "img_size": predictor.img_size,
        "device": str(predictor.device),
        "total_params": total_params,
        "dataset_counts": _dataset_counts_via_polars(),
    }


@app.get("/api/samples")
def api_samples() -> dict[str, object]:
    items: list[dict[str, str]] = []
    if DATASET_DIR.is_dir():
        for cls in CLASS_NAMES:
            try:
                sample_path = _pick_sample_file(cls)
                items.append(
                    {
                        "class_name": cls,
                        "display_label": CLASS_LABELS.get(cls, cls),
                        "filename": sample_path.name,
                    }
                )
            except HTTPException:
                continue
    return {"items": items}


@app.post("/api/predict")
async def api_predict(
    file: Annotated[UploadFile | None, File()] = None,
    sample_class: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.45,
) -> dict[str, object]:
    clamped_alpha = float(min(max(alpha, 0.0), 1.0))
    image = await _load_input_image(file, sample_class)
    predictor = get_predictor()

    t0 = time.perf_counter()
    result = predictor.predict_image(image, alpha=clamped_alpha)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    heat_only = _heatmap_only_image(result.raw_heatmap, image.size)
    probs_payload = {
        cls: {
            "label": CLASS_LABELS.get(cls, cls),
            "prob": prob,
        }
        for cls, prob in result.probabilities.items()
    }

    return {
        "predicted_class": result.predicted_class,
        "display_label": result.display_label,
        "confidence": result.confidence,
        "probabilities": probs_payload,
        "model_mode": result.model_mode,
        "inference_ms": inference_ms,
        "image_width": image.width,
        "image_height": image.height,
        "original_data_url": image_to_data_url(image, fmt="PNG"),
        "heatmap_only_data_url": image_to_data_url(heat_only, fmt="PNG"),
        "overlay_data_url": image_to_data_url(result.heatmap_overlay, fmt="PNG"),
    }


@app.post("/api/report")
async def api_report(
    file: Annotated[UploadFile | None, File()] = None,
    sample_class: Annotated[str | None, Form()] = None,
    alpha: Annotated[float, Form()] = 0.45,
) -> Response:
    clamped_alpha = float(min(max(alpha, 0.0), 1.0))
    image = await _load_input_image(file, sample_class)
    predictor = get_predictor()
    result = predictor.predict_image(image, alpha=clamped_alpha)
    pdf_bytes = create_pdf_report(image, result.heatmap_overlay, result)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="alzheimer_mri_report.pdf"'},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Alzheimer MRI web workbench.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=os.environ.get("MRI_DEVICE", "cpu"))
    args = parser.parse_args()

    os.environ["MRI_DEVICE"] = args.device
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
