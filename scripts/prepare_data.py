"""Dataset notes and optional sample-folder scaffolding.

Holdout rule (read this twice):
    Do NOT train on the official demonstration subset:
      non-AIGC = COCO val2017 (4998)
      AIGC     = DALL·E Advanced (8843)
    That split is for the live demo / iterative scoreboard only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "data" / "samples"


def write_placeholder(path: Path, color: tuple[int, int, int], label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (128, 128), color)
    draw = ImageDraw.Draw(image)
    draw.rectangle((16, 16, 112, 112), outline=(255, 255, 255), width=3)
    draw.text((24, 56), label, fill=(255, 255, 255))
    image.save(path)


def make_samples() -> None:
    write_placeholder(SAMPLE_ROOT / "REAL" / "real_001.png", (40, 90, 50), "REAL")
    write_placeholder(SAMPLE_ROOT / "REAL" / "real_002.png", (50, 80, 40), "REAL")
    write_placeholder(SAMPLE_ROOT / "FAKE" / "fake_001.png", (90, 40, 90), "FAKE")
    write_placeholder(SAMPLE_ROOT / "FAKE" / "fake_002.png", (80, 30, 100), "FAKE")
    print(f"wrote 4 placeholder images under {SAMPLE_ROOT}")
    print("These are NOT a training set. Replace with CIFAKE / SID_Set before real runs.")


def print_sources() -> None:
    print(
        """
Recommended public sources (license them properly before submission):

1. CIFAKE - 32x32, fast to iterate, weak as a sole training set
   https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
   Layout after unzip: train/{REAL,FAKE}  test/{REAL,FAKE}

2. SID_Set - social-media scale, includes real / full_synthetic / tampered
   https://huggingface.co/datasets/saberzl/SID_Set
   For binary AIGC: map real=0, full_synthetic=1, drop or separately study tampered.

3. WildFake - generator-diverse, good for generalization
   https://modelscope.cn/datasets/hy2628982280/WildFake/summary
   Use the translation button on ModelScope before browsing.
   NEVER train on the official demo holdout (COCO val2017 + DALL-E Advanced).

After download, point configs/default.yaml data.train_dir / data.val_dir
at the ImageFolder roots.
""".strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true", help="Write tiny placeholder images")
    args = parser.parse_args()
    print_sources()
    if args.samples:
        make_samples()


if __name__ == "__main__":
    main()
