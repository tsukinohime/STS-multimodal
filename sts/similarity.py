"""Similarity over paired embeddings.

Cosine similarity is the prediction — there is no learned head, no rescaling
and no calibration anywhere in this project.
"""

from __future__ import annotations

import numpy as np


def paired_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise cosine similarity between two ``(n, d)`` matrices.

    Re-normalises defensively: adapters already return unit vectors, but a
    dtype round-trip can leave norms slightly off 1.0.
    """
    emb_a = np.asarray(emb_a, dtype=np.float64)
    emb_b = np.asarray(emb_b, dtype=np.float64)
    if emb_a.shape != emb_b.shape:
        raise ValueError(f"shape mismatch: {emb_a.shape} vs {emb_b.shape}")
    a = emb_a / np.maximum(np.linalg.norm(emb_a, axis=1, keepdims=True), eps)
    b = emb_b / np.maximum(np.linalg.norm(emb_b, axis=1, keepdims=True), eps)
    return np.sum(a * b, axis=1)
