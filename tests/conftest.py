from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def sample_dir(tmp_path: Path) -> Path:
    real = tmp_path / "REAL"
    fake = tmp_path / "FAKE"
    real.mkdir()
    fake.mkdir()
    _box(real / "r1.png", (30, 120, 40), "R")
    _box(real / "r2.png", (40, 100, 50), "R")
    _box(fake / "f1.png", (140, 40, 140), "F")
    _box(fake / "f2.png", (160, 30, 120), "F")
    return tmp_path


def _box(path: Path, color: tuple[int, int, int], text: str) -> None:
    image = Image.new("RGB", (64, 64), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 56, 56), outline=(255, 255, 255))
    draw.text((24, 24), text, fill=(255, 255, 255))
    image.save(path)
