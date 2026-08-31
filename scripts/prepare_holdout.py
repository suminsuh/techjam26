"""Download and extract the official WildFake demonstration holdout subset for zero-shot testing.

Holdout composition:
  - REAL: COCO val2017 (label 0) -> saved to data/wildfake_holdout/REAL
  - FAKE: DALL-E 3 Advanced (label 1) -> saved to data/wildfake_holdout/FAKE

CRITICAL COMPETITION RULE:
  Never train on this folder. Use strictly for zero-shot evaluation and demo benchmarking.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from huggingface_hub import hf_hub_download
import pandas as pd
from PIL import Image
from tqdm import tqdm


REPO_ID = "techjam-aigc/wildfake-eval-subset"


def extract_holdout(
    output_root: str | Path = "data/wildfake_holdout",
    num_shards: int = 2,
    max_per_class: int = 1000,
) -> None:
    root = Path(output_root)
    real_dir = root / "REAL"
    fake_dir = root / "FAKE"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    counts = {"REAL": 0, "FAKE": 0}

    print(f"==> Fetching {num_shards} shard(s) of official demonstration holdout from {REPO_ID}...")
    for shard_idx in range(num_shards):
        if counts["REAL"] >= max_per_class and counts["FAKE"] >= max_per_class:
            break
        filename = f"data/validation-{shard_idx:05d}-of-00008.parquet"
        print(f"Downloading shard {shard_idx + 1}/{num_shards}: {filename}")
        p = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset")
        df = pd.read_parquet(p)

        for idx, row in df.iterrows():
            label = int(row["label"])
            if label == 0:
                if counts["REAL"] >= max_per_class:
                    continue
                target_dir = real_dir
                cls_name = "REAL"
            elif label == 1:
                if counts["FAKE"] >= max_per_class:
                    continue
                target_dir = fake_dir
                cls_name = "FAKE"
            else:
                continue

            raw_img = row["image"]
            img_data = None
            if isinstance(raw_img, dict) and "bytes" in raw_img and raw_img["bytes"] is not None:
                img_data = raw_img["bytes"]
            elif isinstance(raw_img, bytes):
                img_data = raw_img
            elif hasattr(raw_img, "tobytes"):
                img_data = raw_img.tobytes()

            img_id = str(row.get("id", f"holdout_{shard_idx}_{idx}"))
            safe_id = "".join(c for c in img_id if c.isalnum() or c in ("-", "_")).strip()
            filename = f"{safe_id}_{idx}.jpg"
            save_path = target_dir / filename

            try:
                if img_data is not None:
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                else:
                    img = Image.fromarray(raw_img).convert("RGB")
                img.save(save_path, "JPEG", quality=95)
                counts[cls_name] += 1
            except Exception as err:
                print(f"Warning: could not save image {idx}: {err}")

        print(f"Shard {shard_idx + 1} done: {counts['REAL']} REAL (COCO), {counts['FAKE']} FAKE (DALL-E 3)")

    print("\n================ Holdout Extraction Complete ================")
    print(f"REAL (COCO val2017):     {counts['REAL']} images -> {real_dir}")
    print(f"FAKE (DALL-E Advanced):  {counts['FAKE']} images -> {fake_dir}")
    print(f"Total holdout images:    {counts['REAL'] + counts['FAKE']}")
    print("REMINDER: Never train on this folder.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch official WildFake demonstration holdout.")
    parser.add_argument("--shards", type=int, default=2, help="Number of parquet shards to fetch")
    parser.add_argument("--max_per_class", type=int, default=1000, help="Max images per class")
    parser.add_argument("--output_root", type=str, default="data/wildfake_holdout", help="Output directory")
    args = parser.parse_args()

    extract_holdout(
        output_root=args.output_root,
        num_shards=args.shards,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
