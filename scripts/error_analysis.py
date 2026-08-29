from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rift.engine.errors import error_note_from_eval  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the error-analysis note from an eval run.")
    parser.add_argument("--metrics", default="outputs/metrics.json")
    parser.add_argument("--preds", default="outputs/predictions_by_condition.json")
    parser.add_argument("--output", default="outputs/error_analysis.md")
    args = parser.parse_args()
    dest = error_note_from_eval(args.metrics, args.preds, args.output)
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
