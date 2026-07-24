"""Encoder interface + factory.

A ``TextEncoder`` turns a list of strings into an (N, D) float32 embedding
matrix. Backends are swappable so the same evaluation pipeline can compare
different pre-trained encoders.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np


class TextEncoder(ABC):
    name: str

    @abstractmethod
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (len(texts), dim) float32 embedding matrix."""
        raise NotImplementedError


def build_encoder(config) -> TextEncoder:
    """Construct a :class:`TextEncoder` from a :class:`sts.config.Config`."""
    backend = config.encoder_backend.lower()
    if backend in ("sentence-transformers", "st", "sbert"):
        from .sentence_transformer import SentenceTransformerEncoder

        return SentenceTransformerEncoder(
            model_name=config.encoder_name,
            device=config.device,
            batch_size=config.batch_size,
            normalize=config.normalize_embeddings,
            prompt_prefix=config.prompt_prefix,
            max_seq_length=config.max_seq_length,
        )
    if backend in ("hf-mean", "hf", "transformers"):
        from .hf_mean import HFMeanPoolEncoder

        return HFMeanPoolEncoder(
            model_name=config.encoder_name,
            device=config.device,
            batch_size=config.batch_size,
            normalize=config.normalize_embeddings,
            prompt_prefix=config.prompt_prefix,
            max_seq_length=config.max_seq_length,
            pooling=config.hf_pooling,
        )
    raise ValueError(f"unknown encoder_backend: {config.encoder_backend!r}")
