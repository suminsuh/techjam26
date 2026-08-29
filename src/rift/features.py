"""Forensic front-end: residual + frequency cues that survive some, not all, transforms.

JPEG quantization wrecks high-frequency *magnitude* more than phase. Spatial
CNNs that only look at RGB textures collapse after social-media re-encode.
We therefore expose three complementary views to the forensic stream:

1. SRM high-pass residuals — camera / synthesis noise patterns
2. Log FFT magnitude — spectral peaks left by upsamplers and GAN/diffusion steps
3. FFT phase — more stable under JPEG than magnitude (CVPR 2026 phase-robust work)

These are differentiable so the encoder can fine-tune around them.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _srm_kernels() -> torch.Tensor:
    # Three classic 5x5 high-pass / residual filters used in image forensics.
    k1 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    k2 = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, -2, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    k3 = torch.tensor(
        [
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1],
        ],
        dtype=torch.float32,
    ) / 12.0
    kernels = torch.stack([k1, k2, k3], dim=0).unsqueeze(1)
    return kernels


class ForensicFrontend(nn.Module):
    """RGB [B,3,H,W] in [0, 1] -> 9-channel forensic tensor."""

    def __init__(self) -> None:
        super().__init__()
        kernels = _srm_kernels()
        self.register_buffer("srm", kernels, persistent=False)

    def _srm_residual(self, x: torch.Tensor) -> torch.Tensor:
        gray = x.mean(dim=1, keepdim=True)
        residual = F.conv2d(gray, self.srm, padding=2)
        return residual

    def _fft_views(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centered = x - x.mean(dim=(-2, -1), keepdim=True)
        spec = torch.fft.fftshift(torch.fft.fft2(centered, norm="ortho"), dim=(-2, -1))
        mag = torch.log(spec.abs() + 1e-6)
        mag = (mag - mag.mean(dim=(-2, -1), keepdim=True)) / (mag.std(dim=(-2, -1), keepdim=True) + 1e-6)
        phase = spec.angle() / math.pi
        return mag, phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self._srm_residual(x)
        mag, phase = self._fft_views(x)
        return torch.cat([residual, mag, phase], dim=1)
