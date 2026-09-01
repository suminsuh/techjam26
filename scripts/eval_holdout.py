"""Score the official WildFake holdout with a frozen SID_Set threshold.

Never trains. Never refits the cutoff on COCO+DALL-E.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rift.cli import eval_main


def _checkpoint() -> str:
    for candidate in (
        ROOT / "checkpoints" / "sidset" / "best_20k.pt",
        ROOT / "checkpoints" / "sidset" / "best.pt",
        ROOT / "checkpoints" / "best.pt",
    ):
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "No SID_Set checkpoint found. Expected checkpoints/sidset/best_20k.pt"
    )


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    ckpt = _checkpoint()
    print(f"checkpoint={ckpt}", flush=True)
    argv = [
        "--config",
        str(ROOT / "configs" / "holdout.yaml"),
        "--checkpoint",
        ckpt,
        "--output_dir",
        "outputs/holdout",
        "--threshold",
        "0.031700171530246735",
    ]
    eval_main(argv)
