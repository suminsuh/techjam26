from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_error_note(
    paths: list[str],
    labels: list[int],
    scores: list[float],
    threshold: float,
    output_path: str | Path,
    top_k: int = 8,
) -> Path:
    """Rank confident mistakes for the required error-analysis note."""
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    fp = np.where((pred == 1) & (y == 0))[0]
    fn = np.where((pred == 0) & (y == 1))[0]
    fp = fp[np.argsort(-s[fp])[:top_k]]
    fn = fn[np.argsort(s[fn])[:top_k]]

    lines = [
        "# Error analysis note",
        "",
        f"Operating threshold (frozen from clean val): `{threshold:.4f}`",
        "",
        "## False positives (authentic flagged as AI-generated)",
        "",
        "Highest-confidence accusations. Common cases: already-compressed",
        "camera photos, screenshots, or unusually smooth texture.",
        "",
    ]
    for idx in fp:
        lines.append(f"- `{paths[idx]}`  pred={s[idx]:.3f}  label=real")
    if len(fp) == 0:
        lines.append("- none in this split")

    lines += [
        "",
        "## False negatives (AI-generated images missed)",
        "",
        "Lowest-confidence misses. After heavy JPEG or blur the score can",
        "fall below a low-FPR cutoff even when ranking is still strong.",
        "",
    ]
    for idx in fn:
        lines.append(f"- `{paths[idx]}`  pred={s[idx]:.3f}  label=ai-generated")
    if len(fn) == 0:
        lines.append("- none in this split")

    lines += [
        "",
        "## Trade-offs to discuss in the write-up",
        "",
        "- A low-FPR cutoff (default 5%) protects authentic images and will miss",
        "  some generated ones. Do not retune it per transform.",
        "- On the submitted CLIP checkpoint the gate often sits on the spatial",
        "  stream. Do not claim a forensic switch unless the gate columns move.",
        "- SID_Set val is the same generator family as training. Report holdout",
        "  separately with this frozen threshold.",
        "",
    ]
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def error_note_from_eval(metrics_path: str | Path, preds_path: str | Path, output_path: str | Path) -> Path:
    metrics: dict[str, Any] = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    preds: dict[str, Any] = json.loads(Path(preds_path).read_text(encoding="utf-8"))
    clean = preds.get("clean") or next(iter(preds.values()))
    return write_error_note(
        clean["paths"],
        clean["labels"],
        clean["scores"],
        float(metrics["threshold"]),
        output_path,
    )
