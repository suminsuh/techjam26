"""Frozen CLIP spatial encoder (UnivFD-style).

CLIP-ViT-B/32 is ~151M params and stays frozen. Only a small projection
is trained. That is the generalization trick: do not fine-tune CLIP onto
one generator or it overfits CIFAKE/SID_Set artifacts.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CLIP_TIMM = {
    "clip_vit_b32": "vit_base_patch32_clip_224.openai",
    "clip_vit_b16": "vit_base_patch16_clip_224.openai",
    "clip_vit_l14": "vit_large_patch14_clip_224.openai",
}

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class ClipSpatialEncoder(nn.Module):
    def __init__(self, name: str, embed_dim: int) -> None:
        super().__init__()
        import timm

        if name not in CLIP_TIMM:
            known = ", ".join(CLIP_TIMM)
            raise KeyError(f"Unknown CLIP backbone '{name}'. Known: {known}")
        self.backbone = timm.create_model(CLIP_TIMM[name], pretrained=True, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
        feat_dim = int(self.backbone.num_features)
        self.proj = nn.Identity() if feat_dim == embed_dim else nn.Linear(feat_dim, embed_dim)
        self._backbone_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._backbone_grad:
            feats = self.backbone(x)
        else:
            with torch.no_grad():
                feats = self.backbone(x)
        return self.proj(feats.float())
