"""Required TechJam entry point.

Usage:
    python scripts/predict.py --input_dir path/to/images --output predictions.json

Output JSON is a list of {image_path, pred} where pred is P(AI-generated).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rift.cli import predict_main  # noqa: E402

if __name__ == "__main__":
    predict_main()
