"""SentenceTransformers backend.

The generic escape hatch for any encoder published as an SBERT model —
including the tiny MiniLM used to exercise the pipeline without downloading a
multi-GB checkpoint. Pooling and normalisation come from the model's own
``modules.json``; ``prompt_prefix`` covers encoders that expect a literal
instruction prefix (e.g. E5's ``"query: "``).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import ModelSpec, get_logger, torch_dtype
from .base import EncodeOutput, TextEncoder, l2_normalize

_log = get_logger()


class SentenceTransformerEncoder(TextEncoder):
    def __init__(
        self,
        spec: ModelSpec,
        device: str,
        dtype: str,
        batch_size: int,
        max_length: int,
        progress: bool = True,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        self.key = spec.key
        self.model_id = spec.model_id
        self.prompt_prefix = spec.prompt_prefix
        self.truncate_dim = spec.truncate_dim
        self.device = device
        self.dtype_name = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.progress = progress
        self._torch = torch

        _log.info("Loading SentenceTransformer %s on %s / %s", spec.model_id, device, dtype)
        self.model = SentenceTransformer(
            spec.model_id,
            device=device,
            trust_remote_code=spec.trust_remote_code,
            model_kwargs={"dtype": torch_dtype(dtype)},
        )
        self.model.max_seq_length = max_length
        self.model.eval()
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.model.get_sentence_embedding_dimension())
        return self._dim

    def token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.model.tokenizer(
            [self.prompt_prefix + t for t in texts], padding=False, truncation=False
        )["input_ids"]
        return np.array([len(ids) for ids in encoded], dtype=np.int32)

    def encode(self, texts: Sequence[str]) -> EncodeOutput:
        texts = list(texts)
        if not texts:
            return EncodeOutput(
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=bool),
            )

        counts = self.token_lengths(texts)
        prefixed = [self.prompt_prefix + t for t in texts] if self.prompt_prefix else texts
        with self._torch.inference_mode():
            embeddings = self.model.encode(
                prefixed,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=self.progress,
            )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.truncate_dim is not None:
            embeddings = l2_normalize(embeddings[:, : self.truncate_dim])
        return EncodeOutput(
            embeddings=embeddings,
            token_counts=counts,
            truncated=counts > self.max_length,
        )

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": "sentence-transformers",
            "prompt_prefix": self.prompt_prefix,
            "pooling": "from modules.json",
            "normalized_by_model": True,
            "truncate_dim": self.truncate_dim,
            "max_length": self.max_length,
            "dtype": self.dtype_name,
            "device": self.device,
        }

    def close(self) -> None:
        if getattr(self, "model", None) is None:
            return
        from .jina_v5 import _free_device_memory

        del self.model
        self.model = None
        _free_device_memory(self._torch, self.device)
