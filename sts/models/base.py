"""Model-adapter interface and factory.

An adapter wraps one frozen pretrained encoder and exposes a single operation:
strings in, L2-normalised embeddings out, plus the token bookkeeping needed to
report truncation. No adapter trains, fine-tunes or calibrates anything.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from ..config import ModelSpec, RuntimeConfig, resolve_device, resolve_dtype


@dataclass
class EncodeOutput:
    """Embeddings plus per-text token bookkeeping."""

    embeddings: np.ndarray   # (n, dim) float32, unit L2 norm
    token_counts: np.ndarray  # (n,) int32, length before truncation
    truncated: np.ndarray     # (n,) bool

    def __len__(self) -> int:
        return int(self.embeddings.shape[0])


class TextEncoder(ABC):
    """A frozen text encoder."""

    key: str
    model_id: str
    max_length: int

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> EncodeOutput:
        """Return unit-norm embeddings for ``texts``, in order."""

    @abstractmethod
    def token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        """Token count per text *before* truncation, prefix included."""

    @property
    def dim(self) -> int:
        raise NotImplementedError

    def describe(self) -> dict:
        """Adapter details recorded in the run manifest."""
        return {"model_id": self.model_id, "max_length": self.max_length}

    def close(self) -> None:
        """Release device memory. Safe to call twice."""


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def batched(items: Sequence[str], size: int) -> List[Sequence[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_encoder(spec: ModelSpec, runtime: RuntimeConfig) -> TextEncoder:
    """Instantiate the adapter named by ``spec.backend``."""
    device = resolve_device(runtime.device)
    dtype = resolve_dtype(spec.dtype or runtime.dtype, device)
    batch_size = spec.batch_size or runtime.batch_size
    max_length = spec.max_length or runtime.max_length
    backend = spec.backend.strip().lower()

    if backend in ("jina-v5", "jina_v5", "jinav5"):
        from .jina_v5 import JinaV5Encoder

        return JinaV5Encoder(spec, device, dtype, batch_size, max_length, runtime.progress)

    if backend in ("sentence-transformers", "st", "sbert"):
        from .sentence_transformer import SentenceTransformerEncoder

        return SentenceTransformerEncoder(spec, device, dtype, batch_size, max_length, runtime.progress)

    if backend in ("hf", "hf-mean", "transformers"):
        from .hf_mean import HFPooledEncoder

        return HFPooledEncoder(spec, device, dtype, batch_size, max_length, runtime.progress)

    raise ValueError(
        f"unknown backend {spec.backend!r} for model {spec.key!r}; "
        "use jina-v5, sentence-transformers or hf"
    )
