from __future__ import annotations

import argparse
from pathlib import Path

from rift.config import load_config
from rift.engine.evaluate import evaluate_robustness
from rift.engine.predict import predict_directory
from rift.engine.train import train_model


def _add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--checkpoint", default=None, help="Optional .pt checkpoint")


def predict_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score a folder of images. Writes TechJam JSON.")
    _add_shared(parser)
    parser.add_argument("--input_dir", required=True, help="Directory of images (recursed)")
    parser.add_argument("--output", default="predictions.json", help="Output JSON path")
    parser.add_argument("--aux", action="store_true", help="Also write gate weights")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    records = predict_directory(args.input_dir, cfg, checkpoint=args.checkpoint, output_path=args.output, include_aux=args.aux)
    print(f"wrote {len(records)} scores -> {args.output}")


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train RIFT on an ImageFolder REAL/FAKE split.")
    _add_shared(parser)
    parser.add_argument("--train_dir", default=None)
    parser.add_argument("--val_dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.train_dir:
        cfg.setdefault("data", {})["train_dir"] = args.train_dir
    if args.val_dir:
        cfg.setdefault("data", {})["val_dir"] = args.val_dir
    path = train_model(cfg, resume=args.checkpoint)
    print(f"best checkpoint -> {path}")


def eval_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fixed-threshold robustness table.")
    _add_shared(parser)
    parser.add_argument("--data_dir", default=None, help="Labeled ImageFolder to evaluate")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    summary = evaluate_robustness(cfg, data_dir=args.data_dir, checkpoint=args.checkpoint, output_dir=args.output_dir)
    print(f"threshold={summary['threshold']:.4f} (frozen from clean val)")
    for row in summary["table"]:
        print(f"  {row['condition']:16s} acc={row['accuracy']:.3f} auroc={row['auroc']:.3f} fpr={row['fpr']:.3f}")
    dest = Path(args.output_dir or cfg.get("eval", {}).get("output_dir", "outputs"))
    print(f"wrote {dest / 'robustness_table.md'}")


if __name__ == "__main__":
    predict_main()
