"""Correlation metrics for STS evaluation.

Standard STS reporting uses Spearman rank correlation as the headline metric,
with Pearson as a common secondary. Both are computed here; Kendall's tau is
available too but off by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import kendalltau, pearsonr, spearmanr


@dataclass
class Correlations:
    n: int
    spearman: float
    pearson: float
    kendall: Optional[float] = None

    def to_dict(self) -> dict:
        d = {"n": self.n, "spearman": self.spearman, "pearson": self.pearson}
        if self.kendall is not None:
            d["kendall"] = self.kendall
        return d


def compute_correlations(pred: np.ndarray, gold: np.ndarray, with_kendall: bool = False) -> Correlations:
    pred = np.asarray(pred, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    if pred.shape != gold.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gold.shape}")
    spearman = float(spearmanr(pred, gold).statistic)
    pearson = float(pearsonr(pred, gold)[0])
    kendall = float(kendalltau(pred, gold).statistic) if with_kendall else None
    return Correlations(n=int(pred.shape[0]), spearman=spearman, pearson=pearson, kendall=kendall)
