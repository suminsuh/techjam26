from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset

from rift.preprocess import open_rgb, pil_to_tensor
from rift.transforms import OfficialAugment

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FAKE_DIR_NAMES = {"fake", "aigc", "ai", "synthetic", "generated", "full_synthetic"}
REAL_DIR_NAMES = {"real", "non-aigc", "nonaigc", "authentic", "coco"}


def discover_images(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files)


def infer_label(path: Path) -> int | None:
    parts = {part.lower() for part in path.parts}
    if parts & FAKE_DIR_NAMES:
        return 1
    if parts & REAL_DIR_NAMES:
        return 0
    stem = path.parent.name.lower()
    if stem in FAKE_DIR_NAMES:
        return 1
    if stem in REAL_DIR_NAMES:
        return 0
    return None


class FolderDataset(Dataset):
    """ImageFolder-style dataset with optional official-transform training augs."""

    def __init__(
        self,
        root: str | Path,
        image_size: int = 224,
        train: bool = False,
        official_aug_prob: float = 0.0,
        two_view: bool = False,
        transform: Callable | None = None,
    ) -> None:
        self.train = train
        self.image_size = image_size
        self.two_view = two_view
        self.paths = discover_images(root)
        if self.train:
            holdout_signatures = {"coco", "val2017", "dall-e", "dalle", "dalle_advanced"}
            for path in self.paths:
                parts = {part.lower() for part in path.parts}
                if parts & holdout_signatures and "wildfake_holdout" in str(path).lower():
                    raise RuntimeError(f"Attempted to load holdout image for training")
        if not self.paths:
            raise FileNotFoundError(f"No images under {root}")
        self.labels: list[int] = []
        missing = []
        for path in self.paths:
            label = infer_label(path)
            if label is None:
                missing.append(path)
            else:
                self.labels.append(label)
        if missing:
            preview = ", ".join(str(p) for p in missing[:5])
            raise ValueError(
                f"{len(missing)} images have no REAL/FAKE parent folder. Example: {preview}"
            )
        self.train = train
        self.image_size = image_size
        self.two_view = two_view
        self.official = OfficialAugment(probability=official_aug_prob) if train and official_aug_prob > 0 else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        image = open_rgb(path)
        view = self.official(image) if self.official is not None else image
        tensor = self.transform(view) if self.transform is not None else pil_to_tensor(view, self.image_size)
        item = {"image": tensor, "label": self.labels[index], "path": path.as_posix()}
        if self.two_view:
            # Second view is always an official transform so consistency
            # is not a no-op when the first view happened to stay clean.
            second = self.official(image, force=True) if self.official is not None else image
            item["image_aug"] = (
                self.transform(second) if self.transform is not None else pil_to_tensor(second, self.image_size)
            )
        return item


class ImageListDataset(Dataset):
    """Unlabeled directory for the required predict.py contract."""

    def __init__(self, root: str | Path, image_size: int = 224, transform: Callable | None = None) -> None:
        self.paths = discover_images(root)
        if not self.paths:
            raise FileNotFoundError(f"No images under {root}")
        self.image_size = image_size
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        image = open_rgb(path)
        tensor = self.transform(image) if self.transform else pil_to_tensor(image, self.image_size)
        return {"image": tensor, "path": path.as_posix()}
