"""Similarity functions over paired embeddings."""

from __future__ import annotations

import numpy as np


def paired_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Row-wise cosine similarity between two (N, D) embedding matrices.

    Returns a length-N array where element i is cos(emb_a[i], emb_b[i]).
    """
    emb_a = np.asarray(emb_a, dtype=np.float32)
    emb_b = np.asarray(emb_b, dtype=np.float32)
    if emb_a.shape != emb_b.shape:
        raise ValueError(f"shape mismatch: {emb_a.shape} vs {emb_b.shape}")
    a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + eps)
    b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + eps)
    return np.sum(a * b, axis=1)
