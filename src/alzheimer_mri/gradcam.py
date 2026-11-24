"""Pure PyTorch Grad-CAM implementation for ResNet34 MRI explainability."""

from __future__ import annotations

from typing import Self

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn


class GradCAM:
    """Compute class-discriminative Grad-CAM heatmaps using forward/backward hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.layer4[-1]
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(
        self, _module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        self._activations = output.detach()

    def _save_gradient(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        if grad_output[0] is not None:
            self._gradients = grad_output[0].detach()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: int | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        """Return a normalized [H, W] heatmap in [0, 1] and model logits."""
        self.model.zero_grad(set_to_none=True)
        was_training = self.model.training
        self.model.eval()

        with torch.enable_grad():
            logits = self.model(input_tensor)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())
            score = logits[:, class_idx].sum()
            score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        # Global-average-pool gradients across spatial dimensions.
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self._activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_np = cam.squeeze().cpu().numpy().astype(np.float32)

        max_val = float(cam_np.max())
        if max_val > 1e-8:
            cam_np = (cam_np - float(cam_np.min())) / (max_val - float(cam_np.min()) + 1e-8)
        else:
            cam_np = np.zeros_like(cam_np, dtype=np.float32)

        if was_training:
            self.model.train()
        return cam_np, logits.detach()


def render_overlay(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.42,
    cmap_name: str = "turbo",
) -> Image.Image:
    """Blend a normalized [0, 1] heatmap over an RGB PIL image."""
    base = image.convert("RGB")
    heatmap_uint8 = np.clip(heatmap * 255.0, 0, 255).astype(np.uint8)
    resized_mask = Image.fromarray(heatmap_uint8, mode="L").resize(
        base.size, resample=Image.Resampling.BILINEAR
    )
    norm_mask = np.asarray(resized_mask, dtype=np.float32) / 255.0

    colormap = matplotlib.colormaps[cmap_name]
    colored_rgb = (colormap(norm_mask)[..., :3] * 255.0).astype(np.uint8)

    base_arr = np.asarray(base, dtype=np.float32)
    blended = (1.0 - alpha) * base_arr + alpha * colored_rgb.astype(np.float32)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")
