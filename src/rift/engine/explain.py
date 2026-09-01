"""Lightweight Grad-CAM for the demo overlay.

CNN spatial backbones use the last Conv2d. CLIP is a frozen ViT that normally
runs under no_grad, so classic conv CAM never sees a backward — that is the
demo IndexError. For CLIP we enable grad for one forward and CAM the last
transformer block's patch tokens.
"""

from __future__ import annotations

import math
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


def _vit_last_block(module: torch.nn.Module) -> torch.nn.Module | None:
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        backbone = getattr(module, "backbone", None)
        blocks = getattr(backbone, "blocks", None)
    if blocks is None or len(blocks) == 0:
        return None
    return blocks[-1]


def _to_heatmap(cam: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


def _tokens_to_map(tokens: torch.Tensor) -> torch.Tensor:
    """[B, N, C] or [B, N] → [B, 1, H, W], dropping the CLS token when needed."""
    if tokens.ndim == 3:
        tokens = tokens[:, 1:] if tokens.shape[1] > 1 else tokens
        n = tokens.shape[1]
        side = int(math.sqrt(n))
        if side * side != n:
            tokens = tokens[:, : side * side] if side > 0 else tokens
            n = tokens.shape[1]
            side = int(math.sqrt(n))
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
    tokens = tokens[:, 1:] if tokens.shape[1] > 1 else tokens
    n = tokens.shape[1]
    side = int(math.sqrt(n))
    return tokens[:, : side * side].reshape(tokens.shape[0], 1, side, side)


def _run_backward(model, image: torch.Tensor) -> None:
    model.zero_grad(set_to_none=True)
    logit = model(image)
    if isinstance(logit, tuple):
        logit = logit[0]
    logit.reshape(-1).sum().backward()


def _conv_gradcam(model, image: torch.Tensor) -> np.ndarray:
    target = _last_conv(model.spatial)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_mod, _inp, out) -> None:
        activations.append(out)

    def bwd_hook(_mod, _gin, gout) -> None:
        if gout and gout[0] is not None:
            gradients.append(gout[0])

    handles = [
        target.register_forward_hook(fwd_hook),
        target.register_full_backward_hook(bwd_hook),
    ]
    try:
        _run_backward(model, image)
        if not activations or not gradients:
            return _input_saliency(model, image)
        act, grad = activations[0], gradients[0]
        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act).sum(dim=1, keepdim=True))
        return _to_heatmap(cam, image.shape[-2:])
    finally:
        for handle in handles:
            handle.remove()


def _vit_gradcam(model, image: torch.Tensor) -> np.ndarray:
    target = _vit_last_block(model.spatial)
    if target is None:
        return _input_saliency(model, image)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def fwd_hook(_mod, _inp, out) -> None:
        activations.append(out[0] if isinstance(out, tuple) else out)

    def bwd_hook(_mod, _gin, gout) -> None:
        if gout and gout[0] is not None:
            gradients.append(gout[0])

    spatial = model.spatial
    prev = getattr(spatial, "_backbone_grad", False)
    spatial._backbone_grad = True
    handles = [
        target.register_forward_hook(fwd_hook),
        target.register_full_backward_hook(bwd_hook),
    ]
    try:
        _run_backward(model, image)
        if not activations or not gradients:
            return _input_saliency(model, image)
        act_map = _tokens_to_map(activations[0])
        grad_map = _tokens_to_map(gradients[0])
        weights = grad_map.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * act_map).sum(dim=1, keepdim=True))
        return _to_heatmap(cam, image.shape[-2:])
    finally:
        spatial._backbone_grad = prev
        for handle in handles:
            handle.remove()


def _input_saliency(model, image: torch.Tensor) -> np.ndarray:
    """Last-resort map if a stream is detached. Still better than crashing the demo."""
    model.zero_grad(set_to_none=True)
    if not image.requires_grad:
        image = image.detach().requires_grad_(True)
    logit = model(image)
    if isinstance(logit, tuple):
        logit = logit[0]
    logit.reshape(-1).sum().backward()
    if image.grad is None:
        return np.zeros(image.shape[-2:], dtype=np.float32)
    sal = image.grad.detach().abs().mean(dim=1, keepdim=True)
    return _to_heatmap(sal, image.shape[-2:])


def gradcam(model, image: torch.Tensor) -> np.ndarray:
    """image: [1,3,H,W] on the model device. Returns [H,W] heatmap in [0,1]."""
    try:
        model.eval()
        image = image.detach().requires_grad_(True)
        if _vit_last_block(model.spatial) is not None:
            return _vit_gradcam(model, image)
        return _conv_gradcam(model, image)
    except Exception:
        h, w = int(image.shape[-2]), int(image.shape[-1])
        return np.zeros((h, w), dtype=np.float32)


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
        lead = "compression and frequency traces"
    else:
        lead = "how the picture looks"
    return (
        f"Appearance weight {spatial:.2f}, trace weight {forensic:.2f}. "
        f"This call mostly used {lead}."
    )
