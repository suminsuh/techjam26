from __future__ import annotations

import csv
import json
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from rift.data.datasets import FolderDataset, infer_label
from rift.engine.errors import write_error_note
from rift.engine.predict import load_checkpoint
from rift.metrics import choose_threshold, evaluate_scores
from rift.preprocess import open_rgb, pil_to_tensor
from rift.seed import resolve_device
from rift.transforms import get_condition


class ConditionDataset(Dataset):
    """Re-reads source images and applies one official condition before resize."""

    def __init__(self, paths: list[Path], labels: list[int], condition: str, image_size: int) -> None:
        self.paths = paths
        self.labels = labels
        self.condition_name = condition
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        seed = zlib.crc32(path.as_posix().encode("utf-8"))
        image = get_condition(self.condition_name, seed=seed)(open_rgb(path))
        return {
            "image": pil_to_tensor(image, self.image_size),
            "label": self.labels[index],
            "path": path.as_posix(),
        }


@torch.no_grad()
def _score_loader(model, loader, device) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    scores, labels, paths, gates = [], [], [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        logits, aux = model(images, return_aux=True)
        scores.append(torch.sigmoid(logits).cpu().numpy())
        labels.append(np.asarray(batch["label"], dtype=int))
        paths.extend(batch["path"])
        gates.append(aux["gate"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(labels), paths, np.concatenate(gates)


def evaluate_robustness(
    cfg: dict[str, Any],
    data_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    device = resolve_device(cfg.get("device", "auto"))
    image_size = int(cfg.get("image_size", 224))
    eval_cfg = cfg.get("eval", {})
    batch_size = int(eval_cfg.get("batch_size", 16))
    root = Path(data_dir or cfg.get("data", {}).get("val_dir"))
    out_dir = Path(output_dir or eval_cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    base = FolderDataset(root, image_size=image_size, train=False)
    model = load_checkpoint(checkpoint or cfg.get("predict", {}).get("checkpoint"), cfg, device)

    conditions = list(eval_cfg.get("conditions", ["clean"]))
    clean_name = "clean" if "clean" in conditions else conditions[0]

    clean_ds = ConditionDataset(base.paths, base.labels, clean_name, image_size)
    clean_loader = DataLoader(clean_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    clean_scores, clean_labels, _, _ = _score_loader(model, clean_loader, device)
    threshold = choose_threshold(clean_labels, clean_scores, float(eval_cfg.get("target_fpr", 0.05)))

    table: list[dict[str, Any]] = []
    per_condition: dict[str, Any] = {}
    for name in conditions:
        dataset = ConditionDataset(base.paths, base.labels, name, image_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        scores, labels, paths, gates = _score_loader(model, loader, device)
        report = evaluate_scores(labels, scores, threshold)
        row = {"condition": name, **report.as_dict()}
        row["mean_gate_spatial"] = round(float(gates[:, 0].mean()), 4)
        row["mean_gate_forensic"] = round(float(gates[:, 1].mean()), 4)
        table.append(row)
        per_condition[name] = {
            "paths": paths,
            "labels": labels.tolist(),
            "scores": scores.tolist(),
        }

    _write_table(out_dir, table)
    summary = {
        "threshold": threshold,
        "threshold_source": eval_cfg.get("threshold_source", "clean_val"),
        "target_fpr": eval_cfg.get("target_fpr", 0.05),
        "n": int(len(base)),
        "table": table,
        "note": (
            "Threshold was fit on clean images only and frozen for every "
            "transform. Do not retune per condition when reporting robustness."
        ),
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "predictions_by_condition.json").write_text(
        json.dumps({k: v for k, v in per_condition.items()}, indent=2),
        encoding="utf-8",
    )
    clean = per_condition.get("clean") or next(iter(per_condition.values()))
    write_error_note(
        clean["paths"],
        clean["labels"],
        clean["scores"],
        threshold,
        out_dir / "error_analysis.md",
    )
    return summary


def _write_table(out_dir: Path, table: list[dict[str, Any]]) -> None:
    if not table:
        return
    keys = list(table[0].keys())
    with (out_dir / "robustness_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(table)

    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    rows = ["| " + " | ".join(str(row[k]) for k in keys) + " |" for row in table]
    (out_dir / "robustness_table.md").write_text("\n".join([header, sep, *rows]) + "\n", encoding="utf-8")


def labeled_paths(root: str | Path) -> tuple[list[Path], list[int]]:
    files = []
    labels = []
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        label = infer_label(path)
        if label is None:
            continue
        files.append(path)
        labels.append(label)
    return files, labels
