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
import shutil
import zipfile
from PIL import Image, ImageDraw

from rift.data.datasets import discover_images, infer_label

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT/"data"
SAMPLE_ROOT = DATA_ROOT/"samples"
HOLDOUT_ROOT = DATA_ROOT/"wildfake_holdout"


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

def extract_cifake_zip(zip_path: str | Path) -> None:
    """Extracts a downloaded CIFAKE zip into data/cifake/{train,test}/{REAL,FAKE}."""
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found at {zip_path}")
    dest = DATA_ROOT / "cifake"
    print(f"Extracting {zip_path} -> {dest} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    print("CIFAKE extracted successfully.")

def verify_datasets() -> None:
    """Checks all subfolders under data/ for correct layout and holdout isolation."""
    print("=" * 60)
    print("RIFT Dataset Integrity & Holdout Verification")
    print("=" * 60)
    # 1. Check holdout directory
    if HOLDOUT_ROOT.exists():
        holdout_imgs = discover_images(HOLDOUT_ROOT)
        print(f"[*] Found Holdout Directory: {HOLDOUT_ROOT} ({len(holdout_imgs)} images)")
        print("    [!] ENSURE this directory is NEVER set as 'data.train_dir' in configs!")
    else:
        print(f"[-] Holdout directory not yet downloaded at {HOLDOUT_ROOT} (OK for now)")
    # 2. Check each candidate dataset under data/
    candidates = [p for p in DATA_ROOT.iterdir() if p.is_dir() and p.name != "wildfake_holdout"]
    if not candidates:
        print("[-] No dataset folders found under data/. Run with --samples or download datasets.")
        return
    for ds_dir in candidates:
        print(f"\nScanning dataset: {ds_dir.name} ({ds_dir})")
        images = discover_images(ds_dir)
        if not images:
            print("  [x] Empty directory (no images found)")
            continue
        real_count = 0
        fake_count = 0
        unknown = []
        for p in images:
            lbl = infer_label(p)
            if lbl == 0:
                real_count += 1
            elif lbl == 1:
                fake_count += 1
            else:
                unknown.append(p)
        print(f"  - Total images: {len(images)}")
        print(f"  - Real (0):     {real_count}")
        print(f"  - Fake (1):     {fake_count}")
        if unknown:
            print(f"  [!] WARNING: {len(unknown)} images could not be classified into REAL/FAKE!")
            print(f"      Example: {unknown[0]}")
        else:
            print("  [✓] All images successfully matched to REAL/FAKE labels.")
    print("\n" + "=" * 60)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", action="store_true", help="Write tiny placeholder images")
    parser.add_argument("--cifake_zip", type=str, default=None, help="Path to downloaded CIFAKE archive to extract")
    parser.add_argument("--verify", action="store_true", help="Verify folder structures and image labels")
    args = parser.parse_args()
    print_sources()
    if args.samples:
        make_samples()
    if args.cifake_zip:
        extract_cifake_zip(args.cifake_zip)
    if args.verify or (not args.samples and not args.cifake_zip):
        verify_datasets()
                       


if __name__ == "__main__":
    main()
