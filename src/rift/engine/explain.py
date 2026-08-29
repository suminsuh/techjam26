"""Lightweight Grad-CAM over the spatial encoder.

Enough for the error-analysis note and the demo overlay. Not a research CAM
library — keep it dependency-free so the team can run it on CPU.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _last_conv(module: torch.nn.Module) -> torch.nn.Module:
    last = None
    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            last = child
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last


def gradcam(model, image: torch.Tensor) -> np.ndarray:
    """image: [1,3,H,W] on the model device. Returns [H,W] heatmap in [0,1]."""
    model.eval()
    target = _last_conv(model.spatial)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_mod, _inp, out) -> None:
        activations.append(out)

    def bwd_hook(_mod, _gin, gout) -> None:
        gradients.append(gout[0])

    handles = [target.register_forward_hook(fwd_hook), target.register_full_backward_hook(bwd_hook)]
    try:
        model.zero_grad(set_to_none=True)
        logit = model(image)
        logit.sum().backward()
        act = activations[0]
        grad = gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam
    finally:
        for handle in handles:
            handle.remove()


def overlay_heatmap(pil_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    heat = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_image.size, Image.BILINEAR)
    heat_rgb = np.zeros((*heat.size[::-1], 3), dtype=np.uint8)
    heat_arr = np.asarray(heat)
    heat_rgb[..., 0] = heat_arr
    heat_rgb[..., 1] = (heat_arr * 0.3).astype(np.uint8)
    base = np.asarray(pil_image.convert("RGB")).astype(np.float32)
    mixed = (1 - alpha) * base + alpha * heat_rgb.astype(np.float32)
    return Image.fromarray(np.clip(mixed, 0, 255).astype(np.uint8))


def gate_story(gate: torch.Tensor | Any) -> str:
    if hasattr(gate, "detach"):
        values = gate.detach().cpu().flatten().tolist()
    else:
        values = list(gate)
    spatial, forensic = float(values[0]), float(values[1])
    if forensic > spatial:
        lead = "forensic residual / frequency cues"
    else:
        lead = "spatial / semantic texture"
    return (
        f"Trust gate: spatial={spatial:.2f}, forensic={forensic:.2f}. "
        f"This decision leaned on {lead}."
    )
