"""Stream a balanced SID_Set slice to data/sidset/ (real vs full_synthetic)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rift.config import load_config  # noqa: E402
from rift.data.sid_set import ensure_sidset_cache  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "clip_sidset.yaml")
    train_dir, val_dir = ensure_sidset_cache(cfg)
    print(f"train cache -> {train_dir}")
    print(f"val cache   -> {val_dir}")


if __name__ == "__main__":
    main()
