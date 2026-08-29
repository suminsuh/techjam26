"""Official TechJam robustness transforms plus training wrappers.

Parameter values match the Track 5 table exactly. Evaluation must apply these
operations *before* the model resize so we measure redistribution, not our
own preprocessing.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import gaussian_filter


ConditionFn = Callable[[Image.Image], Image.Image]


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def jpeg_compress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    to_rgb(image).save(buffer, format="JPEG", quality=int(quality), optimize=False)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    # scipy uses the spec's σ; PIL's radius is only an approximation.
    array = np.asarray(to_rgb(image), dtype=np.float32)
    blurred = np.stack([gaussian_filter(array[..., c], sigma=float(sigma)) for c in range(3)], axis=-1)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8), mode="RGB")


def down_up_resize(image: Image.Image, scale: float) -> Image.Image:
    image = to_rgb(image)
    width, height = image.size
    small = (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )
    return image.resize(small, Image.BILINEAR).resize((width, height), Image.BILINEAR)


def gaussian_noise(image: Image.Image, sigma: float, seed: int | None = None) -> Image.Image:
    array = np.asarray(to_rgb(image), dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, float(sigma), size=array.shape).astype(np.float32)
    array = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((array * 255.0).astype(np.uint8), mode="RGB")


def color_jitter(image: Image.Image, amplitude: float = 0.20, rng: random.Random | None = None) -> Image.Image:
    rng = rng or random
    image = to_rgb(image)
    factors = {
        "color": rng.uniform(1.0 - amplitude, 1.0 + amplitude),
        "brightness": rng.uniform(1.0 - amplitude, 1.0 + amplitude),
        "contrast": rng.uniform(1.0 - amplitude, 1.0 + amplitude),
    }
    image = ImageEnhance.Color(image).enhance(factors["color"])
    image = ImageEnhance.Brightness(image).enhance(factors["brightness"])
    image = ImageEnhance.Contrast(image).enhance(factors["contrast"])
    return image


def center_crop(image: Image.Image, keep: float = 0.80) -> Image.Image:
    image = to_rgb(image)
    width, height = image.size
    new_w = max(1, int(round(width * keep)))
    new_h = max(1, int(round(height * keep)))
    left = (width - new_w) // 2
    top = (height - new_h) // 2
    return image.crop((left, top, left + new_w, top + new_h))


# name -> factory so evaluate.py can iterate the official grid
CONDITION_FACTORIES: dict[str, Callable[[], ConditionFn]] = {
    "clean": lambda: (lambda img: to_rgb(img)),
    "jpeg_90": lambda: (lambda img: jpeg_compress(img, 90)),
    "jpeg_70": lambda: (lambda img: jpeg_compress(img, 70)),
    "jpeg_50": lambda: (lambda img: jpeg_compress(img, 50)),
    "jpeg_30": lambda: (lambda img: jpeg_compress(img, 30)),
    "blur_0.5": lambda: (lambda img: gaussian_blur(img, 0.5)),
    "blur_1.0": lambda: (lambda img: gaussian_blur(img, 1.0)),
    "blur_2.0": lambda: (lambda img: gaussian_blur(img, 2.0)),
    "resize_0.5": lambda: (lambda img: down_up_resize(img, 0.5)),
    "resize_0.25": lambda: (lambda img: down_up_resize(img, 0.25)),
    "noise_0.02": lambda: (lambda img: gaussian_noise(img, 0.02)),
    "noise_0.05": lambda: (lambda img: gaussian_noise(img, 0.05)),
    "noise_0.10": lambda: (lambda img: gaussian_noise(img, 0.10)),
    "color_jitter": lambda: (lambda img: color_jitter(img, 0.20)),
    "center_crop_0.8": lambda: (lambda img: center_crop(img, 0.80)),
}


@dataclass
class OfficialAugment:
    """Sample one official transform during training (robustness, not just accuracy)."""

    probability: float = 0.7
    exclude_clean: bool = True

    def __post_init__(self) -> None:
        names = [name for name in CONDITION_FACTORIES if not (self.exclude_clean and name == "clean")]
        self._names = names

    def __call__(self, image: Image.Image, force: bool = False) -> Image.Image:
        if not force and random.random() > self.probability:
            return to_rgb(image)
        name = random.choice(self._names)
        return CONDITION_FACTORIES[name]()(image)


def get_condition(name: str, seed: int | None = None) -> ConditionFn:
    if name not in CONDITION_FACTORIES:
        known = ", ".join(sorted(CONDITION_FACTORIES))
        raise KeyError(f"Unknown condition '{name}'. Known: {known}")
    if name.startswith("noise_"):
        sigma = float(name.split("_")[1])
        return lambda img: gaussian_noise(img, sigma, seed=seed)
    if name == "color_jitter":
        rng = random.Random(seed)
        return lambda img: color_jitter(img, 0.20, rng=rng)
    return CONDITION_FACTORIES[name]()
