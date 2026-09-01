from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from rift.data.cifake_hf import CifakeHFDataset
from rift.data.datasets import FolderDataset
from rift.data.sid_set import ensure_sidset_cache
from rift.metrics import choose_threshold, evaluate_scores, safe_auroc
from rift.models.dual_stream import build_model
from rift.seed import resolve_device, seed_everything


def _consistency_loss(clean_logit: torch.Tensor, aug_logit: torch.Tensor) -> torch.Tensor:
    p_clean = torch.sigmoid(clean_logit.detach().float())
    return F.binary_cross_entropy_with_logits(aug_logit.float(), p_clean)


def train_model(cfg: dict[str, Any], resume: str | Path | None = None) -> Path:
    seed_everything(int(cfg.get("seed", 42)))
    device = resolve_device(cfg.get("device", "auto"))
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    image_size = int(cfg.get("image_size", 224))

    max_train = train_cfg.get("max_samples")
    max_val = train_cfg.get("max_val_samples", 4000)
    source = str(data_cfg.get("source", "folder"))
    if source in {"wildfake_holdout", "holdout"}:
        raise RuntimeError(
            "Official WildFake holdout is eval-only. Do not train on COCO val2017 "
            "or DALL-E Advanced. Use SID_Set or CIFAKE instead."
        )
    if source == "cifake_hf":
        train_ds = CifakeHFDataset(
            "train",
            image_size=image_size,
            train=True,
            official_aug_prob=float(train_cfg.get("official_aug_prob", 0.7)),
            two_view=float(train_cfg.get("consistency_weight", 0.5)) > 0,
            max_samples=int(max_train) if max_train else None,
            seed=int(cfg.get("seed", 42)),
        )
        val_ds = CifakeHFDataset(
            "test",
            image_size=image_size,
            train=False,
            max_samples=int(max_val) if max_val else None,
            seed=int(cfg.get("seed", 42)),
        )
    elif source == "sid_set":
        train_dir, val_dir = ensure_sidset_cache(cfg)
        train_ds = FolderDataset(
            train_dir,
            image_size=image_size,
            train=True,
            official_aug_prob=float(train_cfg.get("official_aug_prob", 0.7)),
            two_view=float(train_cfg.get("consistency_weight", 0.5)) > 0,
            max_samples=int(max_train) if max_train else None,
            seed=int(cfg.get("seed", 42)),
        )
        val_ds = FolderDataset(
            val_dir,
            image_size=image_size,
            train=False,
            max_samples=int(max_val) if max_val else None,
            seed=int(cfg.get("seed", 42)),
        )
    else:
        train_dir = Path(data_cfg["train_dir"])
        val_dir = Path(data_cfg["val_dir"])
        if not train_dir.exists():
            raise FileNotFoundError(
                f"Train dir not found: {train_dir}\n"
                "Set data.source: cifake_hf or run python scripts/download_cifake.py"
            )
        if not val_dir.exists():
            raise FileNotFoundError(f"Val dir not found: {val_dir}")
        train_ds = FolderDataset(
            train_dir,
            image_size=image_size,
            train=True,
            official_aug_prob=float(train_cfg.get("official_aug_prob", 0.7)),
            two_view=float(train_cfg.get("consistency_weight", 0.5)) > 0,
            max_samples=int(max_train) if max_train else None,
            seed=int(cfg.get("seed", 42)),
        )
        val_ds = FolderDataset(
            val_dir,
            image_size=image_size,
            train=False,
            max_samples=int(max_val) if max_val else None,
            seed=int(cfg.get("seed", 42)),
        )
    workers = int(train_cfg.get("num_workers", 0))
    if os.name == "nt" and workers > 2:
        workers = 2
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=workers,
        drop_last=False,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
    )

    model = build_model(cfg).to(device)
    if resume:
        payload = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"device={device} train={len(train_ds)} val={len(val_ds)} "
        f"trainable={sum(p.numel() for p in trainable):,} "
        f"total={sum(p.numel() for p in model.parameters()):,}"
    )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    epochs = int(train_cfg.get("epochs", 8))
    patience = train_cfg.get("patience")
    patience = int(patience) if patience is not None else None
    min_delta = float(train_cfg.get("min_delta", 0.001))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    consistency_w = float(train_cfg.get("consistency_weight", 0.5))
    forensic_w = float(train_cfg.get("forensic_aux_weight", 0.0))
    entropy_w = float(train_cfg.get("gate_entropy_weight", 0.0))
    target_fpr = float(cfg.get("eval", {}).get("target_fpr", 0.05))
    save_dir = Path(train_cfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    best_score = -1.0
    best_path = save_dir / "best.pt"
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"train {epoch}/{epochs}"):
            images = batch["image"].to(device, non_blocking=pin)
            labels = batch["label"].float().to(device, non_blocking=pin)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                need_aux = forensic_w > 0 or entropy_w > 0
                if need_aux:
                    logits, aux = model(images, return_aux=True)
                else:
                    logits = model(images)
                    aux = None
                cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = cls_loss
                if forensic_w > 0 and aux is not None:
                    loss = loss + forensic_w * F.binary_cross_entropy_with_logits(
                        aux["forensic_logit"].float(), labels
                    )
                if entropy_w > 0 and aux is not None:
                    gate = aux["gate"].float().clamp_min(1e-8)
                    entropy = -(gate * gate.log()).sum(dim=1).mean()
                    loss = loss - entropy_w * entropy
                if consistency_w > 0 and "image_aug" in batch:
                    aug_logits = model(batch["image_aug"].to(device, non_blocking=pin))
                    loss = loss + consistency_w * _consistency_loss(logits, aug_logits)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += float(loss.detach().cpu())
        scheduler.step()
        val_acc, val_auroc, val_rep = _operating_metrics(model, val_loader, device, target_fpr)
        train_loss = running / max(len(train_loader), 1)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} "
            f"val_auroc={val_auroc:.4f} val_prec={val_rep.precision:.4f} "
            f"val_fpr={val_rep.fpr:.4f} val_fnr={val_rep.fnr:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )
        score = val_acc if val_auroc == val_auroc else val_acc
        payload = {
            "model": model.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "val_auroc": val_auroc,
            "val_precision": val_rep.precision,
            "val_fpr": val_rep.fpr,
            "val_fnr": val_rep.fnr,
            "config": cfg,
            "param_count": model.param_count(),
        }
        torch.save(payload, save_dir / "last.pt")
        if score > best_score + min_delta:
            best_score = score
            stale = 0
            torch.save(payload, best_path)
            print(f"  saved {best_path} (score={score:.4f})", flush=True)
        else:
            if score >= best_score:
                best_score = score
                torch.save(payload, best_path)
                print(f"  saved {best_path} (tie score={score:.4f})", flush=True)
            stale += 1
            if patience is not None:
                print(f"  no operating-point gain (stale={stale}/{patience})", flush=True)
                if stale >= patience:
                    print(f"early stop at epoch {epoch} (best val acc@FPR={best_score:.4f})", flush=True)
                    break

    return best_path


@torch.no_grad()
def _operating_metrics(model, loader, device, target_fpr: float):
    model.eval()
    scores, labels = [], []
    for batch in loader:
        probs = torch.sigmoid(model(batch["image"].to(device)))
        scores.append(probs.detach().cpu().numpy())
        labels.append(np.asarray(batch["label"], dtype=int))
    y_score = np.concatenate(scores)
    y_true = np.concatenate(labels)
    threshold = choose_threshold(y_true, y_score, target_fpr)
    report = evaluate_scores(y_true, y_score, threshold)
    return report.accuracy, safe_auroc(y_true, y_score), report
