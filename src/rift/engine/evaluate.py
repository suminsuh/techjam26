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

from rift.data.cifake_hf import CifakeHFConditionDataset
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
    raw_root = data_dir or cfg.get("data", {}).get("val_dir")
    root = Path(raw_root) if raw_root else Path(".")
    out_dir = Path(output_dir or eval_cfg.get("output_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    max_samples = eval_cfg.get("max_samples")
    source = str(cfg.get("data", {}).get("source", "folder"))
    model = load_checkpoint(checkpoint or cfg.get("predict", {}).get("checkpoint"), cfg, device)

    conditions = list(eval_cfg.get("conditions", ["clean"]))
    clean_name = "clean" if "clean" in conditions else conditions[0]
    seed = int(cfg.get("seed", 42))
    n_limit = int(max_samples) if max_samples else None

    cached_clean: tuple[np.ndarray, np.ndarray, list[str], np.ndarray] | None = None

    if source == "cifake_hf":
        base_len = n_limit or 2000
        def _make_condition(name: str):
            return CifakeHFConditionDataset("test", name, image_size, n_limit, seed)
    elif source == "sid_set":
        from rift.data.sid_set import ensure_sidset_cache

        _, val_dir = ensure_sidset_cache(cfg)
        root = val_dir
        base = FolderDataset(root, image_size=image_size, train=False, max_samples=n_limit, seed=seed)
        base_len = len(base)
        def _make_condition(name: str):
            return ConditionDataset(base.paths, base.labels, name, image_size)
    elif source in {"wildfake_holdout", "holdout"}:
        from rift.data.holdout import HoldoutConditionDataset, _indices, load_holdout

        holdout_config = str(cfg.get("data", {}).get("holdout_config", "default"))
        holdout_ds = load_holdout(holdout_config)
        clean_indices = _indices(holdout_ds, n_limit, seed, balanced=False)
        robust_n = eval_cfg.get("robustness_samples")
        if robust_n:
            robust_indices = _indices(holdout_ds, int(robust_n), seed, balanced=True)
        else:
            robust_indices = clean_indices
        base_len = len(clean_indices)

        def _make_condition(name: str):
            idxs = clean_indices if name == clean_name else robust_indices
            return HoldoutConditionDataset(
                holdout_config,
                name,
                image_size,
                None,
                seed,
                ds=holdout_ds,
                indices=idxs,
            )
    else:
        base = FolderDataset(root, image_size=image_size, train=False, max_samples=n_limit, seed=seed)
        base_len = len(base)
        def _make_condition(name: str):
            return ConditionDataset(base.paths, base.labels, name, image_size)

    frozen = eval_cfg.get("frozen_threshold")
    if source in {"wildfake_holdout", "holdout"} and frozen is None:
        raise ValueError(
            "Holdout eval must reuse a frozen threshold from SID_Set or CIFAKE val. "
            "Set eval.frozen_threshold (SID_Set 20k = 0.0317). Do not fit on holdout."
        )
    if frozen is not None:
        threshold = float(frozen)
        threshold_source = "frozen"
    else:
        clean_loader = DataLoader(_make_condition(clean_name), batch_size=batch_size, shuffle=False, num_workers=0)
        cached_clean = _score_loader(model, clean_loader, device)
        threshold = choose_threshold(cached_clean[1], cached_clean[0], float(eval_cfg.get("target_fpr", 0.05)))
        threshold_source = str(eval_cfg.get("threshold_source", "clean_val"))

    table: list[dict[str, Any]] = []
    per_condition: dict[str, Any] = {}
    for name in conditions:
        if cached_clean is not None and name == clean_name:
            scores, labels, paths, gates = cached_clean
        else:
            dataset = _make_condition(name)
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
        "threshold_source": threshold_source,
        "target_fpr": eval_cfg.get("target_fpr", 0.05),
        "n": int(base_len),
        "data_source": source,
        "holdout_config": str(cfg.get("data", {}).get("holdout_config", "")) or None,
        "table": table,
        "note": (
            "Threshold was frozen for every transform. Holdout never retunes "
            "the cutoff. Do not fit a new operating point on COCO+DALL-E."
            if source in {"wildfake_holdout", "holdout"}
            else (
                "Threshold was fit on clean images only and frozen for every "
                "transform. Do not retune per condition when reporting robustness."
            )
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
