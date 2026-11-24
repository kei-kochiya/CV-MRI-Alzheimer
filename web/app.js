(() => {
  "use strict";

  const state = {
    currentFile: null,
    currentSampleClass: null,
    viewMode: "overlay",
    opacity: 0.45,
    busy: false,
  };

  const els = {
    statusPill: document.getElementById("model-status-pill"),
    statusText: document.getElementById("model-status-text"),
    previewNotice: document.getElementById("preview-notice"),
    dropzone: document.getElementById("dropzone"),
    dropzoneTitle: document.getElementById("dropzone-title"),
    dropzoneHint: document.getElementById("dropzone-hint"),
    fileInput: document.getElementById("mri-file-input"),
    samplesGrid: document.getElementById("samples-grid"),
    clearBtn: document.getElementById("clear-btn"),
    feedback: document.getElementById("feedback-line"),
    diagBlock: document.getElementById("diagnosis-block"),
    diagStage: document.getElementById("diag-stage"),
    diagConfidence: document.getElementById("diag-confidence"),
    probList: document.getElementById("prob-list"),
    downloadPdfBtn: document.getElementById("download-pdf-btn"),
    rerunBtn: document.getElementById("rerun-btn"),
    viewOverlayBtn: document.getElementById("view-overlay-btn"),
    viewSplitBtn: document.getElementById("view-split-btn"),
    opacityInput: document.getElementById("heatmap-opacity"),
    opacityValue: document.getElementById("opacity-value"),
    scanGrid: document.getElementById("scan-grid"),
    imgPrimaryBase: document.getElementById("img-primary-base"),
    imgPrimaryHeat: document.getElementById("img-primary-heat"),
    emptyPrimary: document.getElementById("empty-primary"),
    captionPrimaryLeft: document.getElementById("caption-primary-left"),
    captionPrimaryRight: document.getElementById("caption-primary-right"),
    secondaryFrame: document.getElementById("secondary-frame"),
    imgSecondaryHeat: document.getElementById("img-secondary-heat"),
    telArch: document.getElementById("tel-arch"),
    telDevice: document.getElementById("tel-device"),
    telLatency: document.getElementById("tel-latency"),
    telValAcc: document.getElementById("tel-valacc"),
    specParams: document.getElementById("spec-params"),
  };

  function setFeedback(text, kind = "idle") {
    els.feedback.textContent = text;
    els.feedback.dataset.state = kind;
  }

  function setBusy(isBusy, activeLabel = null) {
    state.busy = isBusy;
    els.dropzone.dataset.state = isBusy ? "loading" : "default";
    els.rerunBtn.disabled = isBusy || (!state.currentFile && !state.currentSampleClass);
    els.downloadPdfBtn.disabled = isBusy || (!state.currentFile && !state.currentSampleClass);
    const sampleButtons = els.samplesGrid.querySelectorAll("button");
    sampleButtons.forEach((btn) => {
      btn.disabled = isBusy;
    });
    if (isBusy && activeLabel) {
      setFeedback(activeLabel, "loading");
    }
  }

  function updateViewLayout() {
    const isSplit = state.viewMode === "side-by-side";
    els.scanGrid.dataset.view = state.viewMode;
    els.viewOverlayBtn.setAttribute("aria-pressed", String(!isSplit));
    els.viewSplitBtn.setAttribute("aria-pressed", String(isSplit));

    const hasImage = !els.imgPrimaryBase.hidden;
    els.secondaryFrame.hidden = !isSplit || !hasImage;
    els.imgPrimaryHeat.hidden = isSplit || !hasImage;
    els.imgPrimaryHeat.style.opacity = isSplit ? "0" : String(state.opacity);
    els.captionPrimaryLeft.textContent = isSplit ? "Original axial MRI" : "Axial MRI + Grad-CAM";
  }

  function renderProbabilities(probabilities, topClass) {
    els.probList.innerHTML = "";
    const entries = Object.entries(probabilities).sort((a, b) => b[1].prob - a[1].prob);

    entries.forEach(([clsKey, item]) => {
      const pct = (item.prob * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "prob-row" + (clsKey === topClass ? " is-top" : "");

      const meta = document.createElement("div");
      meta.className = "prob-row__meta";

      const nameSpan = document.createElement("span");
      nameSpan.textContent = item.label;

      const valSpan = document.createElement("span");
      valSpan.className = "mono-num";
      valSpan.textContent = `${pct}%`;

      meta.append(nameSpan, valSpan);

      const track = document.createElement("div");
      track.className = "prob-track";

      const fill = document.createElement("div");
      fill.className = "prob-fill";
      fill.style.width = `${Math.max(0, Math.min(100, item.prob * 100)).toFixed(1)}%`;

      track.appendChild(fill);
      row.append(meta, track);
      els.probList.appendChild(row);
    });
  }

  function renderPrediction(data) {
    els.diagBlock.hidden = false;
    els.clearBtn.hidden = false;
    els.dropzone.dataset.state = "success";

    els.diagStage.textContent = data.display_label;
    els.diagStage.dataset.severity = data.predicted_class === "NonDemented" ? "healthy" : "impaired";
    els.diagConfidence.textContent = `${(data.confidence * 100).toFixed(2)}%`;

    renderProbabilities(data.probabilities, data.predicted_class);

    els.emptyPrimary.hidden = true;
    els.imgPrimaryBase.src = data.original_data_url;
    els.imgPrimaryBase.hidden = false;

    els.imgPrimaryHeat.src = data.heatmap_only_data_url;
    els.imgSecondaryHeat.src = data.overlay_data_url;
    els.imgSecondaryHeat.hidden = false;

    els.captionPrimaryRight.textContent = `${data.image_width} × ${data.image_height} px`;
    els.telLatency.textContent = `${data.inference_ms.toFixed(1)} ms`;

    updateViewLayout();
    setFeedback("", "idle");
  }

  async function runPrediction({ file = null, sampleClass = null }) {
    if (state.busy) return;

    const formData = new FormData();
    if (file) {
      formData.append("file", file);
    } else if (sampleClass) {
      formData.append("sample_class", sampleClass);
    } else {
      return;
    }
    formData.append("alpha", String(state.opacity));

    setBusy(true, "Running ResNet34 forward pass and Grad-CAM…");
    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || "The server could not process this image.");
      }
      renderPrediction(payload);
    } catch (err) {
      els.dropzone.dataset.state = "error";
      setFeedback(
        `Scan failed: ${err.message} Upload a valid JPEG or PNG MRI slice and try again.`,
        "error"
      );
    } finally {
      setBusy(false);
    }
  }

  async function downloadPdfReport() {
    if (state.busy || (!state.currentFile && !state.currentSampleClass)) return;

    const formData = new FormData();
    if (state.currentFile) {
      formData.append("file", state.currentFile);
    } else if (state.currentSampleClass) {
      formData.append("sample_class", state.currentSampleClass);
    }
    formData.append("alpha", String(state.opacity));

    els.downloadPdfBtn.dataset.state = "loading";
    els.downloadPdfBtn.textContent = "Exporting…";
    setBusy(true);
    try {
      const response = await fetch("/api/report", { method: "POST", body: formData });
      if (!response.ok) {
        let msg = "PDF generation failed.";
        try {
          const errPayload = await response.json();
          if (errPayload.detail) msg = errPayload.detail;
        } catch (_) {}
        throw new Error(msg);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "alzheimer_mri_report.pdf";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);

      els.downloadPdfBtn.dataset.state = "success";
      setFeedback("", "idle");
    } catch (err) {
      els.downloadPdfBtn.dataset.state = "error";
      setFeedback(`PDF export failed: ${err.message} Run the scan again and retry.`, "error");
    } finally {
      els.downloadPdfBtn.textContent = "Export PDF";
      setBusy(false);
    }
  }

  function handleSelectedFile(file) {
    if (!file) return;
    const validTypes = ["image/jpeg", "image/png"];
    const validExt = /\.(jpe?g|png)$/i.test(file.name);
    if (!validTypes.includes(file.type) && !validExt) {
      els.dropzone.dataset.state = "error";
      setFeedback(
        "Unsupported file type. Only JPEG and PNG axial MRI images are supported — choose a .jpg or .png file.",
        "error"
      );
      return;
    }

    state.currentFile = file;
    state.currentSampleClass = null;
    els.dropzoneTitle.textContent = file.name;
    els.dropzoneHint.textContent = `${(file.size / 1024).toFixed(1)} KB · ready`;
    els.samplesGrid.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));
    runPrediction({ file });
  }

  function resetWorkbench() {
    state.currentFile = null;
    state.currentSampleClass = null;
    els.fileInput.value = "";
    els.dropzone.dataset.state = "default";
    els.dropzoneTitle.textContent = "Select or drop an MRI image";
    els.dropzoneHint.textContent = "JPEG or PNG axial slice — resized to 224 × 224 px";
    els.samplesGrid.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", "false"));

    els.diagBlock.hidden = true;
    els.clearBtn.hidden = true;
    els.imgPrimaryBase.hidden = true;
    els.imgPrimaryHeat.hidden = true;
    els.imgSecondaryHeat.hidden = true;
    els.secondaryFrame.hidden = true;
    els.emptyPrimary.hidden = false;
    els.telLatency.textContent = "—";
    els.captionPrimaryRight.textContent = "layer4[-1]";
    setFeedback("", "idle");
    updateViewLayout();
  }

  async function loadInitialState() {
    try {
      const [statusRes, samplesRes] = await Promise.all([
        fetch("/api/status"),
        fetch("/api/samples"),
      ]);
      const status = await statusRes.json();
      const samples = await samplesRes.json();

      const isTrained = status.model_mode === "trained";
      els.statusPill.dataset.mode = status.model_mode;
      els.statusText.textContent = isTrained
        ? `Trained · ${(status.val_acc * 100).toFixed(1)}% val`
        : "Preview · untrained weights";
      els.previewNotice.hidden = isTrained;

      els.telArch.textContent = status.arch;
      els.telDevice.textContent = status.device.toUpperCase();
      els.telValAcc.textContent = isTrained ? `${(status.val_acc * 100).toFixed(2)}%` : "Pending";
      if (els.specParams && status.total_params) {
        els.specParams.textContent = `torchvision.models.resnet34 (${status.total_params.toLocaleString()} params)`;
      }

      if (status.dataset_counts) {
        Object.entries(status.dataset_counts).forEach(([cls, count]) => {
          const cell = document.querySelector(`[data-cls="${cls}"]`);
          if (cell) cell.textContent = `${count.toLocaleString()} scans`;
        });
      }

      els.samplesGrid.innerHTML = "";
      if (samples.items && samples.items.length > 0) {
        samples.items.forEach((item) => {
          const btn = document.createElement("button");
          btn.type = "button";
          btn.className = "btn btn--sample";
          btn.setAttribute("aria-pressed", "false");
          btn.dataset.sampleClass = item.class_name;

          const labelSpan = document.createElement("span");
          labelSpan.textContent = item.display_label;

          btn.appendChild(labelSpan);
          btn.addEventListener("click", () => {
            state.currentFile = null;
            state.currentSampleClass = item.class_name;
            els.fileInput.value = "";
            els.dropzoneTitle.textContent = `Sample: ${item.filename}`;
            els.dropzoneHint.textContent = `${item.display_label} preset from dataset/`;
            els.samplesGrid
              .querySelectorAll("button")
              .forEach((b) => b.setAttribute("aria-pressed", b === btn ? "true" : "false"));
            runPrediction({ sampleClass: item.class_name });
          });

          els.samplesGrid.appendChild(btn);
        });
      } else {
        const emptyNote = document.createElement("span");
        emptyNote.className = "samples__label";
        emptyNote.textContent = "No dataset/ samples found — upload a scan above.";
        els.samplesGrid.appendChild(emptyNote);
      }
    } catch (err) {
      els.statusText.textContent = "Backend unreachable";
      setFeedback("Could not load model status. Start the FastAPI server and reload.", "error");
    }
  }

  // Event wiring
  els.fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) handleSelectedFile(file);
  });

  els.dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    els.dropzone.classList.add("is-dragover");
  });

  els.dropzone.addEventListener("dragleave", () => {
    els.dropzone.classList.remove("is-dragover");
  });

  els.dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    els.dropzone.classList.remove("is-dragover");
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleSelectedFile(file);
  });

  els.viewOverlayBtn.addEventListener("click", () => {
    state.viewMode = "overlay";
    updateViewLayout();
  });

  els.viewSplitBtn.addEventListener("click", () => {
    state.viewMode = "side-by-side";
    updateViewLayout();
  });

  els.opacityInput.addEventListener("input", (e) => {
    const pct = Number(e.target.value);
    state.opacity = pct / 100;
    els.opacityValue.textContent = `${pct}%`;
    if (state.viewMode === "overlay") {
      els.imgPrimaryHeat.style.opacity = String(state.opacity);
    }
  });

  els.rerunBtn.addEventListener("click", () => {
    runPrediction({ file: state.currentFile, sampleClass: state.currentSampleClass });
  });

  els.downloadPdfBtn.addEventListener("click", downloadPdfReport);
  els.clearBtn.addEventListener("click", resetWorkbench);

  updateViewLayout();
  loadInitialState();
})();
