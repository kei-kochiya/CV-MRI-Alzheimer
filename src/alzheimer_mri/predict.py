"""Single-image ResNet34 inference, Grad-CAM explainability, and PDF report export."""

from __future__ import annotations

import argparse
import base64
import datetime
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fpdf import FPDF
from PIL import Image

from alzheimer_mri.config import CLASS_LABELS, CLASS_NAMES
from alzheimer_mri.data import get_transforms
from alzheimer_mri.gradcam import GradCAM, render_overlay
from alzheimer_mri.model import build_resnet34, load_checkpoint, resolve_device


@dataclass(slots=True)
class PredictionResult:
    predicted_class: str
    display_label: str
    confidence: float
    probabilities: dict[str, float]
    heatmap_overlay: Image.Image
    raw_heatmap: np.ndarray
    model_mode: str  # "trained" or "preview_untrained"


def image_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """Encode a PIL image to a base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def create_pdf_report(
    original_image: Image.Image,
    overlay_image: Image.Image,
    result: PredictionResult,
) -> bytes:
    """Generate a clinical summary PDF containing original MRI, Grad-CAM, and scores."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "ALZHEIMER MRI SCREENING REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", size=10)
    timestamp = datetime.datetime.now(datetime.UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    pdf.cell(0, 7, f"Generated: {timestamp} | Backbone: PyTorch ResNet34 ({result.model_mode})", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, f"Predicted Stage: {result.display_label}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 9, f"Confidence: {result.confidence * 100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Class Probability Distribution:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=10)
    for cls_name, prob in result.probabilities.items():
        label = CLASS_LABELS.get(cls_name, cls_name)
        pdf.cell(0, 6, f"  - {label}: {prob * 100:.2f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    tmp_orig = None
    tmp_cam = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f1:
            original_image.convert("RGB").save(f1.name)
            tmp_orig = f1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f2:
            overlay_image.convert("RGB").save(f2.name)
            tmp_cam = f2.name

        y_top = pdf.get_y()
        pdf.image(tmp_orig, x=15, y=y_top, w=82)
        pdf.image(tmp_cam, x=110, y=y_top, w=82)
        pdf.set_y(y_top + 86)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(95, 6, "Original Axial MRI", align="C")
        pdf.cell(95, 6, "ResNet34 Grad-CAM Attention Overlay", align="C", new_x="LMARGIN", new_y="NEXT")
    finally:
        for path in (tmp_orig, tmp_cam):
            if path and os.path.exists(path):
                os.unlink(path)

    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(
        0,
        5,
        "Disclaimer: This automated deep-learning output is for research and educational "
        "decision support only and does not constitute a medical diagnosis.",
    )
    return bytes(pdf.output())


class AlzheimerPredictor:
    """Thread-safe inference wrapper for ResNet34 + Grad-CAM."""

    def __init__(
        self,
        checkpoint_path: Path = Path("artifacts/resnet34_best.pt"),
        device_name: str = "auto",
        allow_untrained_fallback: bool = True,
    ):
        self.device = resolve_device(device_name)
        self.checkpoint_path = Path(checkpoint_path)

        if self.checkpoint_path.is_file():
            self.model, meta = load_checkpoint(self.checkpoint_path, device=self.device)
            self.class_names = tuple(meta.get("class_names", CLASS_NAMES))
            self.img_size = int(meta.get("img_size", 224))
            self.val_acc = float(meta.get("val_acc", 0.0))
            self.epoch = int(meta.get("epoch", 0))
            self.model_mode = "trained"
        elif allow_untrained_fallback:
            self.model = build_resnet34(num_classes=len(CLASS_NAMES), pretrained=False).to(self.device)
            self.model.eval()
            self.class_names = CLASS_NAMES
            self.img_size = 224
            self.val_acc = 0.0
            self.epoch = 0
            self.model_mode = "preview_untrained"
        else:
            raise FileNotFoundError(
                f"Trained ResNet34 checkpoint not found at {self.checkpoint_path}. "
                "Run `python -m alzheimer_mri.train` first."
            )

        self.transform = get_transforms(img_size=self.img_size, is_train=False)

    def predict_image(self, image: Image.Image, alpha: float = 0.42) -> PredictionResult:
        rgb_image = image.convert("RGB")
        tensor = self.transform(rgb_image).unsqueeze(0).to(self.device)

        with GradCAM(self.model) as cam:
            heatmap, logits = cam.generate(tensor)

        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
        best_idx = int(np.argmax(probs))
        best_class = self.class_names[best_idx]
        overlay = render_overlay(rgb_image, heatmap, alpha=alpha)

        prob_map = {cls: float(probs[i]) for i, cls in enumerate(self.class_names)}
        return PredictionResult(
            predicted_class=best_class,
            display_label=CLASS_LABELS.get(best_class, best_class),
            confidence=float(probs[best_idx]),
            probabilities=prob_map,
            heatmap_overlay=overlay,
            raw_heatmap=heatmap,
            model_mode=self.model_mode,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Alzheimer's stage for an MRI image.")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/resnet34_best.pt"))
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = AlzheimerPredictor(checkpoint_path=args.checkpoint, device_name=args.device)
    with Image.open(args.image) as img:
        res = predictor.predict_image(img)
    print(f"Mode: {res.model_mode}")
    print(f"Prediction: {res.display_label} ({res.confidence * 100:.2f}%)")
    for cls, prob in res.probabilities.items():
        print(f"  {CLASS_LABELS.get(cls, cls)}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
