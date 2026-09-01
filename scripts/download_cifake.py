"""Download the official CIFAKE split from Hugging Face into ImageFolder layout.

Label note: the HF mirror stores 0=FAKE, 1=REAL. We write folder names so
RIFT's convention stays pred = P(AIGC) = FAKE.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def export_split(ds, dest: Path, label_names: dict[int, str]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(tqdm(ds, desc=f"write {dest.name}", total=len(ds))):
        name = label_names[int(row["label"])]
        folder = dest / name
        folder.mkdir(parents=True, exist_ok=True)
        image = row["image"].convert("RGB")
        image.save(folder / f"{idx:06d}.jpg", quality=95)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/cifake")
    args = parser.parse_args()
    from datasets import load_dataset

    print("loading dragonintelligence/CIFAKE-image-dataset ...")
    bundle = load_dataset("dragonintelligence/CIFAKE-image-dataset")
    raw_names = bundle["train"].features["label"].names
    names = {i: str(n).upper() for i, n in enumerate(raw_names)}
    print("label map", names)
    out = Path(args.out)
    export_split(bundle["train"], out / "train", names)
    export_split(bundle["test"], out / "test", names)
    print(f"wrote CIFAKE under {out.resolve()}")


if __name__ == "__main__":
    main()
