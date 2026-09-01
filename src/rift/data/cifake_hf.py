"""CIFAKE via the Hugging Face mirror (already cached after one download).

HF labels: 0=FAKE, 1=REAL. RIFT labels: 1=AIGC/FAKE, 0=REAL.
"""

from __future__ import annotations

import random
import zlib
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

from rift.preprocess import pil_to_tensor
from rift.transforms import OfficialAugment, get_condition, to_rgb


def _load_split(split: str):
    from datasets import load_dataset

    return load_dataset("dragonintelligence/CIFAKE-image-dataset", split=split)


def _to_rift_label(hf_label: int) -> int:
    return 0 if int(hf_label) == 1 else 1


def _balanced_indices(ds, max_samples: int | None, seed: int) -> list[int]:
    n = len(ds)
    if max_samples is None or max_samples >= n:
        return list(range(n))
    rng = random.Random(seed)
    raw = ds["label"]
    buckets: dict[int, list[int]] = {0: [], 1: []}
    for i, hf_label in enumerate(raw):
        buckets[_to_rift_label(int(hf_label))].append(i)
    per = max(1, max_samples // 2)
    chosen: list[int] = []
    for pool in buckets.values():
        rng.shuffle(pool)
        chosen.extend(pool[:per])
    rng.shuffle(chosen)
    return chosen


class CifakeHFDataset(Dataset):
    def __init__(
        self,
        split: str,
        image_size: int = 224,
        train: bool = False,
        official_aug_prob: float = 0.0,
        two_view: bool = False,
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        self.ds = _load_split(split)
        self.indices = _balanced_indices(self.ds, max_samples, seed)
        self.image_size = image_size
        self.two_view = two_view
        self.official = OfficialAugment(probability=official_aug_prob) if train and official_aug_prob > 0 else None
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def _row(self, index: int) -> tuple[Image.Image, int, str]:
        row = self.ds[self.indices[index]]
        image = to_rgb(row["image"])
        label = _to_rift_label(int(row["label"]))
        path = f"cifake/{self.split}/{self.indices[index]:06d}.jpg"
        return image, label, path

    def __getitem__(self, index: int) -> dict[str, Any]:
        image, label, path = self._row(index)
        view = self.official(image) if self.official is not None else image
        item = {
            "image": pil_to_tensor(view, self.image_size),
            "label": label,
            "path": path,
        }
        if self.two_view:
            second = self.official(image, force=True) if self.official is not None else image
            item["image_aug"] = pil_to_tensor(second, self.image_size)
        return item


class CifakeHFConditionDataset(Dataset):
    def __init__(self, split: str, condition: str, image_size: int, max_samples: int | None, seed: int) -> None:
        self.ds = _load_split(split)
        self.indices = _balanced_indices(self.ds, max_samples, seed)
        self.condition_name = condition
        self.image_size = image_size
        self.split = split

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.ds[self.indices[index]]
        image = to_rgb(row["image"])
        label = _to_rift_label(int(row["label"]))
        path = f"cifake/{self.split}/{self.indices[index]:06d}.jpg"
        seed = zlib.crc32(path.encode("utf-8"))
        image = get_condition(self.condition_name, seed=seed)(image)
        return {"image": pil_to_tensor(image, self.image_size), "label": label, "path": path}
