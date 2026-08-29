from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rift.data.datasets import FolderDataset
from rift.models.dual_stream import build_model
from rift.seed import resolve_device, seed_everything


def _consistency_loss(clean_logit: torch.Tensor, aug_logit: torch.Tensor) -> torch.Tensor:
    """Keep the AIGC score stable after a second official transform.

    This is the training-time version of the problem statement: a reposted
    image should not flip the decision. We match probabilities, not logits,
    so the loss stays in the same units as the submitted `pred` scores.
    """
    p_clean = torch.sigmoid(clean_logit.detach())
    p_aug = torch.sigmoid(aug_logit)
    return F.binary_cross_entropy(p_aug, p_clean)


def train_model(cfg: dict[str, Any], resume: str | Path | None = None) -> Path:
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    image_size = int(cfg.get("image_size", 224))

    train_dir = Path(data_cfg["train_dir"])
    val_dir = Path(data_cfg["val_dir"])
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Train dir not found: {train_dir}\n"
            "Download CIFAKE or SID_Set first. Never point this at the WildFake holdout."
        )
    if not val_dir.exists():
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    train_ds = FolderDataset(
        train_dir,
        image_size=image_size,
        train=True,
        official_aug_prob=float(train_cfg.get("official_aug_prob", 0.7)),
        two_view=float(train_cfg.get("consistency_weight", 0.5)) > 0,
    )
    val_ds = FolderDataset(val_dir, image_size=image_size, train=False)
    workers = 0 if os.name == "nt" else int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=workers,
    )

    model = build_model(cfg).to(device)
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    consistency_w = float(train_cfg.get("consistency_weight", 0.5))
    save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_path = save_dir / "best.pt"
    epochs = int(train_cfg.get("epochs", 8))

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch}/{epochs}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = cls_loss
                if consistency_w > 0 and "image_aug" in batch:
                    aug_logits = model(batch["image_aug"].to(device))
                    loss = cls_loss + consistency_w * _consistency_loss(logits, aug_logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
        val_acc = _quick_accuracy(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running / max(len(train_loader), 1):.4f} val_acc={val_acc:.4f}")
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "config": cfg,
            "param_count": model.param_count(),
        }
        torch.save(payload, save_dir / "last.pt")
        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(payload, best_path)

    return best_path


@torch.no_grad()
def _quick_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        probs = torch.sigmoid(model(batch["image"].to(device)))
        pred = (probs >= 0.5).long()
        correct += int((pred.cpu() == batch["label"]).sum())
        total += int(pred.numel())
    return correct / max(total, 1)
