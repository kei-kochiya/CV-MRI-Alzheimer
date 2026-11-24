# Alzheimer's Stage Detection via MRI (PyTorch ResNet34 + Polars)

Deep learning pipeline and diagnostic web workbench for classifying axial brain MRI scans into four stages (`NonDemented`, `VeryMildDemented`, `MildDemented`, `ModerateDemented`) with ResNet34 and Grad-CAM explainability.

## Project Structure

- `src/alzheimer_mri/config.py`: Class labels and `TrainConfig` hyperparameters.
- `src/alzheimer_mri/data.py`: Polars dataset scanner, deterministic stratified `70 / 15 / 15` split builder (`artifacts/manifest.parquet`), class-weight computation, and PyTorch `DataLoader`s.
- `src/alzheimer_mri/model.py`: `torchvision.models.resnet34` builder, backbone freeze/unfreeze helpers, and checkpoint save/load.
- `src/alzheimer_mri/gradcam.py`: Pure PyTorch Grad-CAM targeting `model.layer4[-1]`.
- `src/alzheimer_mri/train.py`: Two-stage training CLI (frozen head warmup + full fine-tuning with AMP, `AdamW`, cosine schedule, early stopping, and Polars metric logging).
- `src/alzheimer_mri/evaluate.py`: Test-split evaluator generating Polars predictions (`artifacts/test_predictions.parquet`) and `artifacts/confusion_matrix.png`.
- `src/alzheimer_mri/predict.py`: Single-image inference, Grad-CAM blending, and PDF report export.
- `app.py` + `web/` + `tokens.css`: FastAPI web workbench with drag-and-drop upload, dataset sample presets, live Grad-CAM opacity slider, side-by-side comparison, and PDF export.
- `tests/test_pipeline.py`: CPU unit and FastAPI smoke tests.

## Setup

```bash
uv venv .venv --system-site-packages
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe --no-deps -e .
```

## Usage

### 1. Run CPU Verification Tests

```bash
.venv/Scripts/python.exe -m pytest -v
```

### 2. Train ResNet34 (When GPU Is Free)

```bash
.venv/Scripts/python.exe -m alzheimer_mri.train --data-dir dataset --output-dir artifacts --batch-size 32 --img-size 224
```

Optional quick smoke run on CPU:

```bash
.venv/Scripts/python.exe -m alzheimer_mri.train --dry-run --device cpu --no-pretrained --output-dir artifacts_dryrun
```

### 3. Evaluate on Held-Out Test Split

```bash
.venv/Scripts/python.exe -m alzheimer_mri.evaluate --checkpoint artifacts/resnet34_best.pt
```

### 4. Launch the Web Workbench

```bash
.venv/Scripts/python.exe app.py --host 127.0.0.1 --port 8000 --device cpu
```

Then open `http://127.0.0.1:8000` in your browser.
