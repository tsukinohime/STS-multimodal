"""Correlation metrics.

Spearman's rho is the headline number (standard for STS and invariant to any
monotone rescaling of cosine, which matters because nothing here is
calibrated); Pearson's r is reported alongside it. The optional pair-level
bootstrap resamples pairs with replacement to give a percentile 95% CI.

Datasets are never concatenated before correlating — a global correlation over
pooled corpora mostly measures the between-corpus score offsets. Each dataset
(and each domain within it) is scored on its own and combined with an
*unweighted* macro average.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr


@dataclass
class Correlations:
    n: int
    spearman: float
    pearson: float
    spearman_ci: Optional[tuple] = None
    pearson_ci: Optional[tuple] = None
    n_bootstrap: int = 0

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "n": self.n,
            "spearman": self.spearman,
            "pearson": self.pearson,
        }
        if self.spearman_ci is not None:
            payload["spearman_ci_low"], payload["spearman_ci_high"] = self.spearman_ci
            payload["pearson_ci_low"], payload["pearson_ci_high"] = self.pearson_ci
            payload["n_bootstrap"] = self.n_bootstrap
        return payload


def _safe_corr(fn, pred: np.ndarray, gold: np.ndarray) -> float:
    """Correlation that returns NaN instead of raising on degenerate input.

    A constant vector (possible in a tiny bootstrap resample or a one-item
    domain) has undefined correlation.
    """
    if pred.size < 2 or np.all(pred == pred[0]) or np.all(gold == gold[0]):
        return float("nan")
    value = fn(pred, gold)
    statistic = value.statistic if hasattr(value, "statistic") else value[0]
    return float(statistic)


def compute_correlations(
    pred: Sequence[float],
    gold: Sequence[float],
    bootstrap: int = 0,
    seed: int = 12345,
    confidence: float = 0.95,
) -> Correlations:
    pred = np.asarray(pred, dtype=np.float64)
    gold = np.asarray(gold, dtype=np.float64)
    if pred.shape != gold.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {gold.shape}")

    spearman = _safe_corr(spearmanr, pred, gold)
    pearson = _safe_corr(pearsonr, pred, gold)

    spearman_ci = pearson_ci = None
    if bootstrap and pred.size >= 2:
        rng = np.random.default_rng(seed)
        rho_samples = np.empty(bootstrap)
        r_samples = np.empty(bootstrap)
        n = pred.size
        for i in range(bootstrap):
            index = rng.integers(0, n, size=n)
            rho_samples[i] = _safe_corr(spearmanr, pred[index], gold[index])
            r_samples[i] = _safe_corr(pearsonr, pred[index], gold[index])
        alpha = (1.0 - confidence) / 2.0
        spearman_ci = _percentile_ci(rho_samples, alpha)
        pearson_ci = _percentile_ci(r_samples, alpha)

    return Correlations(
        n=int(pred.size),
        spearman=spearman,
        pearson=pearson,
        spearman_ci=spearman_ci,
        pearson_ci=pearson_ci,
        n_bootstrap=int(bootstrap),
    )


def _percentile_ci(samples: np.ndarray, alpha: float) -> Optional[tuple]:
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return None
    low, high = np.percentile(finite, [100 * alpha, 100 * (1 - alpha)])
    return (float(low), float(high))


def macro_average(values: Iterable[float]) -> float:
    """Unweighted mean over datasets, ignoring NaNs."""
    finite = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.mean(finite)) if finite else float("nan")
