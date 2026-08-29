from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rift.config import PredictItem
from rift.data.datasets import ImageListDataset, discover_images
from rift.models.dual_stream import DualStreamDetector, build_model
from rift.preprocess import open_rgb, pil_to_tensor
from rift.seed import resolve_device
from rift.transforms import center_crop, jpeg_compress


def load_checkpoint(path: str | Path | None, cfg: dict[str, Any], device: torch.device) -> DualStreamDetector:
    model = build_model(cfg)
    resolved = Path(path) if path else None
    if resolved and resolved.exists():
        payload = torch.load(resolved, map_location=device, weights_only=False)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state)
    else:
        location = str(resolved) if resolved else "(none)"
        warnings.warn(
            f"No checkpoint at {location}. Using randomly initialized weights. "
            "These scores are meaningless — train first, then pass --checkpoint.",
            stacklevel=2,
        )
    model.to(device)
    model.eval()
    return model


def _tta_views(path: str | Path, image_size: int) -> torch.Tensor:
    image = open_rgb(path)
    views = [image, jpeg_compress(image, 90), center_crop(image, 0.8)]
    return torch.stack([pil_to_tensor(view, image_size) for view in views])


@torch.no_grad()
def predict_directory(
    input_dir: str | Path,
    cfg: dict[str, Any],
    checkpoint: str | Path | None = None,
    output_path: str | Path | None = None,
    include_aux: bool = False,
) -> list[dict[str, Any]]:
    device = resolve_device(cfg.get("device", "auto"))
    image_size = int(cfg.get("image_size", 224))
    batch_size = int(cfg.get("predict", {}).get("batch_size", 16))
    tta = bool(cfg.get("predict", {}).get("tta", False))
    model = load_checkpoint(checkpoint or cfg.get("predict", {}).get("checkpoint"), cfg, device)

    items: list[PredictItem] = []
    if tta:
        paths = discover_images(input_dir)
        for path in tqdm(paths, desc="predict", leave=False):
            views = _tta_views(path, image_size).to(device)
            logits, aux = model(views, return_aux=True)
            pred = float(torch.sigmoid(logits).mean().item())
            gate = aux["gate"].mean(dim=0).cpu()
            items.append(
                PredictItem(
                    image_path=path.as_posix(),
                    pred=pred,
                    extras={"gate_spatial": float(gate[0]), "gate_forensic": float(gate[1])},
                )
            )
    else:
        dataset = ImageListDataset(input_dir, image_size=image_size)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for batch in tqdm(loader, desc="predict", leave=False):
            images = batch["image"].to(device)
            logits, aux = model(images, return_aux=True)
            probs = torch.sigmoid(logits).detach().cpu()
            gates = aux["gate"].detach().cpu()
            for path, pred, gate in zip(batch["path"], probs, gates, strict=True):
                items.append(
                    PredictItem(
                        image_path=str(path),
                        pred=float(pred),
                        extras={"gate_spatial": float(gate[0]), "gate_forensic": float(gate[1])},
                    )
                )

    records = []
    for item in items:
        record = item.to_required()
        if include_aux:
            record.update(item.extras)
        records.append(record)

    if output_path:
        dest = Path(output_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records
