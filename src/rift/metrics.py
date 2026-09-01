from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)


@dataclass
class BinaryReport:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    auroc: float
    ap: float
    fpr: float
    fnr: float
    balanced_acc: float
    n: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": round(self.threshold, 4),
            "accuracy": round(self.accuracy, 4),
            "balanced_acc": round(self.balanced_acc, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
            "f1": round(self.f1, 4),
            "auroc": round(self.auroc, 4),
            "ap": round(self.ap, 4),
            "fpr": round(self.fpr, 4),
            "fnr": round(self.fnr, 4),
            "n": int(self.n),
        }


def safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, target_fpr: float = 0.05) -> float:
    """Pick one operating point on *clean* validation and freeze it.

    Platforms cannot retune the cutoff for every JPEG quality an image hit
    on the way to the feed. We therefore prefer a low-FPR point (false
    accusation is worse for creators) and reuse it on every transform.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, _, thresholds = roc_curve(y_true, y_score)
    finite = np.isfinite(thresholds)
    fpr, thresholds = fpr[finite], thresholds[finite]
    if len(thresholds) == 0:
        return 0.5
    eligible = np.where(fpr <= target_fpr)[0]
    if len(eligible) == 0:
        return float(thresholds[int(np.argmin(np.abs(fpr - target_fpr)))])
    return float(thresholds[eligible[-1]])


def evaluate_scores(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> BinaryReport:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_hat = (y_score >= threshold).astype(int)
    fp = int(np.sum((y_hat == 1) & (y_true == 0)))
    fn = int(np.sum((y_hat == 0) & (y_true == 1)))
    tn = int(np.sum((y_hat == 0) & (y_true == 0)))
    tp = int(np.sum((y_hat == 1) & (y_true == 1)))
    fpr = fp / max(fp + tn, 1)
    fnr = fn / max(fn + tp, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return BinaryReport(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_hat)),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
        f1=float(f1_score(y_true, y_hat, zero_division=0)),
        auroc=safe_auroc(y_true, y_score),
        ap=safe_ap(y_true, y_score),
        fpr=float(fpr),
        fnr=float(fnr),
        balanced_acc=float((recall + specificity) / 2.0),
        n=int(len(y_true)),
    )
