"""SID_Set cache: real vs full_synthetic only.

The Hugging Face dump is ~140GB. We pull one parquet shard at a time,
drop tampered (label 2 — mostly real pixels, not image-level AIGC),
downscale to max side 512, and write an ImageFolder cache. Each shard
is deleted after use so a 4070 laptop with ~50GB free can scale toward
the public 100k-scale split without keeping the full dump.
"""

from __future__ import annotations

import io
import json
import re
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

from rift.transforms import to_rgb

HF_ID = "saberzl/SID_Set"
REAL_LABEL = 0
FAKE_LABEL = 1
TAMPERED_LABEL = 2

CLASS_DIR = {REAL_LABEL: "REAL", FAKE_LABEL: "FAKE"}
SPLIT_FILES = {
    "train": ("train", 249),
    "validation": ("validation", 34),
}
# First cache run streamed this many examples; resume shards after that.
BOOTSTRAP_NEXT_SHARD = {"train": 36, "validation": 7}


def _safe_id(raw: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(raw)).strip("_")
    return (text or "img")[:80]


def _downscale(image: Image.Image, max_side: int) -> Image.Image:
    image = to_rgb(image)
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(size, Image.BILINEAR)


def _count_jpegs(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})


def _pil_from_cell(cell: Any) -> Image.Image | None:
    if cell is None:
        return None
    if isinstance(cell, Image.Image):
        return cell
    if isinstance(cell, dict):
        raw = cell.get("bytes")
        if raw:
            return Image.open(io.BytesIO(raw))
        nested = cell.get("path")
        if nested and Path(str(nested)).exists():
            return Image.open(nested)
    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell)))
    return None


def _progress_path(dest: Path) -> Path:
    return dest / ".stream_progress.json"


def _load_progress(dest: Path, split: str, have: dict[int, int]) -> int:
    path = _progress_path(dest)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        return int(payload.get("next_shard", 0))
    if have[REAL_LABEL] > 0 or have[FAKE_LABEL] > 0:
        return int(BOOTSTRAP_NEXT_SHARD.get(split, 0))
    return 0


def _save_progress(dest: Path, next_shard: int) -> None:
    _progress_path(dest).write_text(json.dumps({"next_shard": next_shard}), encoding="utf-8")


def _materialize_split(
    split: str,
    dest: Path,
    per_class: int,
    max_side: int,
    cache_dir: Path,
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for name in CLASS_DIR.values():
        (dest / name).mkdir(parents=True, exist_ok=True)

    have = {label: _count_jpegs(dest / name) for label, name in CLASS_DIR.items()}
    if all(count >= per_class for count in have.values()):
        print(f"sid_set {split}: cache already filled ({have})", flush=True)
        return

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    file_stem, n_shards = SPLIT_FILES[split]
    shard = _load_progress(dest, split, have)
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"sid_set {split}: shards from {shard}/{n_shards} until {per_class}/class "
        f"(have {have})",
        flush=True,
    )

    while shard < n_shards and not all(have[k] >= per_class for k in CLASS_DIR):
        filename = f"data/{file_stem}-{shard:05d}-of-{n_shards:05d}.parquet"
        local = None
        last_err: Exception | None = None
        for attempt in range(1, 6):
            try:
                local = hf_hub_download(
                    HF_ID,
                    filename,
                    repo_type="dataset",
                    cache_dir=str(cache_dir),
                )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 — CDN/XET flakes; retry then fail
                last_err = exc
                print(f"  shard {shard} download failed attempt {attempt}/5: {exc}", flush=True)
                shutil.rmtree(cache_dir, ignore_errors=True)
                cache_dir.mkdir(parents=True, exist_ok=True)
        if local is None:
            raise RuntimeError(f"Failed to download {filename}") from last_err
        meta = pq.read_table(local, columns=["img_id", "label"])
        img_ids = meta.column("img_id").to_pylist()
        labels = [int(v) for v in meta.column("label").to_pylist()]
        needed: list[int] = []
        for index, (raw_id, label) in enumerate(zip(img_ids, labels, strict=True)):
            if label not in CLASS_DIR or have[label] >= per_class:
                continue
            out = dest / CLASS_DIR[label] / f"{_safe_id(raw_id)}.jpg"
            if out.exists():
                continue
            needed.append(index)

        if needed:
            images = pq.read_table(local, columns=["image"]).column("image")
            for index in needed:
                label = labels[index]
                if have[label] >= per_class:
                    continue
                try:
                    image = _pil_from_cell(images[index].as_py())
                    if image is None:
                        continue
                    image = _downscale(image, max_side)
                except (OSError, ValueError, TypeError):
                    continue
                out = dest / CLASS_DIR[label] / f"{_safe_id(img_ids[index])}.jpg"
                if out.exists():
                    continue
                image.save(out, format="JPEG", quality=90, optimize=True)
                have[label] += 1
                if have[label] % 1000 == 0:
                    print(
                        f"  {split}: real={have[REAL_LABEL]} fake={have[FAKE_LABEL]} "
                        f"shard={shard}",
                        flush=True,
                    )
                if all(have[k] >= per_class for k in CLASS_DIR):
                    break

        shard += 1
        _save_progress(dest, shard)
        try:
            Path(local).unlink(missing_ok=True)
        except OSError:
            pass
        shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"sid_set {split}: done real={have[REAL_LABEL]} fake={have[FAKE_LABEL]} "
        f"next_shard={shard}",
        flush=True,
    )
    if not all(have[k] >= per_class for k in CLASS_DIR):
        raise RuntimeError(
            f"SID_Set {split} ended before filling the cache: {have} (need {per_class}/class)"
        )


def ensure_sidset_cache(cfg: dict[str, Any]) -> tuple[Path, Path]:
    data_cfg = cfg.get("data", {})
    root = Path(data_cfg.get("cache_dir", "data/sidset"))
    train_n = int(data_cfg.get("train_per_class", 10000))
    val_n = int(data_cfg.get("val_per_class", 2000))
    max_side = int(data_cfg.get("cache_max_side", 512))
    stream_cache = root.parent / ".hf_sidset_stream"
    train_dir = root / "train"
    val_dir = root / "val"
    _materialize_split("train", train_dir, train_n, max_side, stream_cache)
    _materialize_split("validation", val_dir, val_n, max_side, stream_cache)
    shutil.rmtree(stream_cache, ignore_errors=True)
    return train_dir, val_dir
