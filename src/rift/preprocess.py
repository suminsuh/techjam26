"""Shared image -> tensor path. Always [0, 1] RGB.

ImageNet mean/std is applied inside the model, and only on the spatial
stream. The forensic front-end must see raw [0, 1] pixels or FFT/SRM
cues are junk.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
from torchvision import transforms

from rift.transforms import to_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def open_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as handle:
        return to_rgb(handle).copy()


def pil_to_tensor(image: Image.Image, image_size: int):
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return tfm(to_rgb(image))
