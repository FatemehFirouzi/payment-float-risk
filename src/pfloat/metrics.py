"""Model-risk metrics used in development and monitoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

EPS = 1e-4


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def ks(y: np.ndarray, score: np.ndarray) -> float:
    """Max distance between the cumulative bad and good score distributions."""
    fpr, tpr, _ = roc_curve(y, score)
    return float(np.max(tpr - fpr))


def gini(y: np.ndarray, score: np.ndarray) -> float:
    return 2.0 * auc(y, score) - 1.0


def decile_cutoffs(scores: np.ndarray) -> np.ndarray:
    """Nine cutoffs from a baseline distribution, frozen for later comparison."""
    return np.quantile(np.asarray(scores, dtype=float), np.linspace(0.1, 0.9, 9))


def assign_decile(scores: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
    """1 = lowest risk ... 10 = highest risk, using fixed cutoffs."""
    return np.searchsorted(cutoffs, np.asarray(scores, dtype=float), side="left") + 1


def psi(expected_share: np.ndarray, actual_share: np.ndarray) -> float:
    e = np.clip(np.asarray(expected_share, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual_share, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_from_scores(baseline: np.ndarray, current: np.ndarray, cutoffs: np.ndarray | None = None) -> float:
    cutoffs = decile_cutoffs(baseline) if cutoffs is None else cutoffs
    b = np.bincount(assign_decile(baseline, cutoffs), minlength=11)[1:]
    c = np.bincount(assign_decile(current, cutoffs), minlength=11)[1:]
    return psi(b / b.sum(), c / c.sum())


def summary(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "n": len(y),
        "bad_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "auc": auc(y, p),
        "ks": ks(y, p),
        "gini": gini(y, p),
        "brier": float(np.mean((p - y) ** 2)),
    }


def rank_order_table(y: np.ndarray, score: np.ndarray, cutoffs: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"y": y, "score": score, "decile": assign_decile(score, cutoffs)})
    out = d.groupby("decile").agg(n=("y", "size"), bads=("y", "sum"), bad_rate=("y", "mean"),
                                  min_score=("score", "min"), max_score=("score", "max"))
    return out.reset_index()
