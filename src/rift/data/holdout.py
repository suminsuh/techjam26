"""Official WildFake demonstration holdout. Eval only — never train.

COCO val2017 (4998 real) + DALL-E Advanced (8843 AIGC). Labels already
match RIFT: 0=real, 1=AIGC. Prefer the TechJam parquet mirror; fall back
is documented in the error if the org dataset is private.
"""

from __future__ import annotations

import random
import zlib
from typing import Any

from torch.utils.data import Dataset

from rift.preprocess import pil_to_tensor
from rift.transforms import get_condition, to_rgb

HF_ID = "techjam-aigc/wildfake-eval-subset"


def load_holdout(config: str = "default"):
    from datasets import load_dataset

    try:
        return load_dataset(HF_ID, config, split="validation")
    except Exception as exc:
        raise RuntimeError(
            f"Could not load {HF_ID} ({config}). That repo is private to the "
            "techjam-aigc org — run `hf auth login` after an invite. "
            "Do not train on this set. Original error: "
            f"{exc}"
        ) from exc


def _indices(ds, max_samples: int | None, seed: int, balanced: bool) -> list[int]:
    n = len(ds)
    if max_samples is None or max_samples >= n:
        return list(range(n))
    rng = random.Random(seed)
    if not balanced:
        chosen = list(range(n))
        rng.shuffle(chosen)
        return chosen[:max_samples]
    buckets: dict[int, list[int]] = {0: [], 1: []}
    for i, label in enumerate(ds["label"]):
        buckets[int(label)].append(i)
    per = max(1, max_samples // 2)
    chosen: list[int] = []
    for pool in buckets.values():
        rng.shuffle(pool)
        chosen.extend(pool[:per])
    rng.shuffle(chosen)
    return chosen


class HoldoutConditionDataset(Dataset):
    def __init__(
        self,
        config: str,
        condition: str,
        image_size: int,
        max_samples: int | None,
        seed: int,
        balanced: bool = False,
        ds=None,
        indices: list[int] | None = None,
    ) -> None:
        self.ds = ds if ds is not None else load_holdout(config)
        self.indices = indices if indices is not None else _indices(self.ds, max_samples, seed, balanced)
        self.condition_name = condition
        self.image_size = image_size
        self.config = config

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[self.indices[index]]
        image = to_rgb(row["image"])
        label = int(row["label"])
        path = str(row.get("id") or f"holdout/{self.config}/{self.indices[index]:06d}")
        seed = zlib.crc32(path.encode("utf-8"))
        image = get_condition(self.condition_name, seed=seed)(image)
        return {"image": pil_to_tensor(image, self.image_size), "label": label, "path": path}
