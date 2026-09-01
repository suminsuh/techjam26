"""Dual-stream AIGC detector with an explainable trust gate.

Why two streams
---------------
Frequency / residual CNNs are accurate on clean images and then fall apart
after JPEG and blur (forensic cues get quantized away). Semantic backbones
degrade more slowly because they look at texture regularity and structure.
A learned gate lets the model *say* which evidence it used — useful in the
demo and in error analysis.

Parameter budget stays far under the <2B rule: tiny ≈ 0.4M, EfficientNet-B0
≈ 5M + forensic head, ConvNeXt-Tiny ≈ 28M + forensic head.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rift.features import ForensicFrontend
from rift.models.clip_stream import CLIP_MEAN, CLIP_STD, ClipSpatialEncoder
from rift.preprocess import IMAGENET_MEAN, IMAGENET_STD


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyEncoder(nn.Module):
    """Small CNN used for the forensic stream and the default spatial stream."""

    def __init__(self, in_ch: int, embed_dim: int) -> None:
        super().__init__()
        self.stem = ConvBlock(in_ch, 32, stride=2)
        self.stages = nn.Sequential(
            ConvBlock(32, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, embed_dim, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stages(self.stem(x))
        return self.pool(x).flatten(1)


class TimmSpatialEncoder(nn.Module):
    def __init__(self, name: str, embed_dim: int, pretrained: bool) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0)
        feat_dim = int(self.backbone.num_features)
        self.proj = nn.Identity() if feat_dim == embed_dim else nn.Linear(feat_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        return self.proj(feats)


class GatedFusion(nn.Module):
    """Softmax gate over (spatial, forensic). Returned for the demo / analysis."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 2),
        )

    def forward(self, spatial: torch.Tensor, forensic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.gate(torch.cat([spatial, forensic], dim=1)), dim=1)
        fused = torch.cat([weights[:, :1] * spatial, weights[:, 1:] * forensic], dim=1)
        return fused, weights


class DualStreamDetector(nn.Module):
    def __init__(
        self,
        spatial_backbone: str = "tiny",
        embed_dim: int = 256,
        pretrained: bool = False,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.spatial_backbone = spatial_backbone
        self.normalize_spatial = spatial_backbone != "tiny"
        self.frontend = ForensicFrontend()
        self.forensic = TinyEncoder(in_ch=9, embed_dim=embed_dim)
        if spatial_backbone == "tiny":
            self.spatial = TinyEncoder(in_ch=3, embed_dim=embed_dim)
        elif spatial_backbone.startswith("clip_"):
            self.spatial = ClipSpatialEncoder(spatial_backbone, embed_dim)
        else:
            self.spatial = TimmSpatialEncoder(spatial_backbone, embed_dim, pretrained)
        mean = CLIP_MEAN if spatial_backbone.startswith("clip_") else IMAGENET_MEAN
        std = CLIP_STD if spatial_backbone.startswith("clip_") else IMAGENET_STD
        self.register_buffer("pixel_mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("pixel_std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)
        self.fusion = GatedFusion(embed_dim)
        self.forensic_head = nn.Linear(embed_dim, 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        forensic = self.forensic(self.frontend(x))
        spatial_x = (x - self.pixel_mean) / self.pixel_std if self.normalize_spatial else x
        spatial = self.spatial(spatial_x)
        fused, gate = self.fusion(spatial, forensic)
        logit = self.head(fused).squeeze(-1)
        if return_aux:
            forensic_logit = self.forensic_head(forensic).squeeze(-1)
            return logit, {
                "gate": gate,
                "spatial": spatial,
                "forensic": forensic,
                "forensic_logit": forensic_logit,
            }
        return logit

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

    def param_count(self) -> int:
        return count_parameters(self)


def build_model(cfg: dict[str, Any]) -> DualStreamDetector:
    model_cfg = cfg.get("model", {})
    return DualStreamDetector(
        spatial_backbone=model_cfg.get("spatial_backbone", "tiny"),
        embed_dim=int(model_cfg.get("embed_dim", 256)),
        pretrained=bool(model_cfg.get("pretrained", False)),
        dropout=float(model_cfg.get("dropout", 0.2)),
    )
