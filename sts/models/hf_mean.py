"""Raw ``transformers`` backend with mean- or CLS-pooling.

For encoders that are not shipped as SentenceTransformers models. Mean pooling
is attention-masked so padding never contributes. This is the backend the
E5 family needs when run without the SBERT wrapper: official E5 usage is
average pooling over the last hidden state plus the ``"query: "`` prefix on
every text for symmetric tasks such as STS.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..config import ModelSpec, get_logger, torch_dtype
from .base import EncodeOutput, TextEncoder, batched, l2_normalize

_log = get_logger()


class HFPooledEncoder(TextEncoder):
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
        from transformers import AutoModel, AutoTokenizer

        if spec.pooling not in ("mean", "cls"):
            raise ValueError(f"{spec.key}: pooling must be 'mean' or 'cls', got {spec.pooling!r}")

        self.key = spec.key
        self.model_id = spec.model_id
        self.prompt_prefix = spec.prompt_prefix
        self.pooling = spec.pooling
        self.truncate_dim = spec.truncate_dim
        self.device = device
        self.dtype_name = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.progress = progress
        self._torch = torch

        _log.info("Loading %s on %s / %s (pooling=%s)", spec.model_id, device, dtype, spec.pooling)
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            spec.model_id, trust_remote_code=spec.trust_remote_code, dtype=torch_dtype(dtype)
        )
        self.model = self.model.to(device).eval()
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.model.config.hidden_size)
        return self._dim

    def token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        encoded = self.tokenizer(
            [self.prompt_prefix + t for t in texts], padding=False, truncation=False
        )["input_ids"]
        return np.array([len(ids) for ids in encoded], dtype=np.int32)

    def _pool(self, hidden, attention_mask):
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def encode(self, texts: Sequence[str]) -> EncodeOutput:
        torch = self._torch
        texts = list(texts)
        if not texts:
            return EncodeOutput(
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                np.zeros((0,), dtype=bool),
            )

        counts = self.token_lengths(texts)
        prefixed = [self.prompt_prefix + t for t in texts] if self.prompt_prefix else texts

        batches = batched(prefixed, self.batch_size)
        iterator = batches
        if self.progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(batches, desc=f"encode[{self.key}]", unit="batch", leave=False)
            except ImportError:
                pass

        chunks: List[np.ndarray] = []
        with torch.inference_mode():
            for batch in iterator:
                encoded = self.tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                pooled = self._pool(hidden, encoded["attention_mask"])
                chunks.append(pooled.float().cpu().numpy())

        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        if self.truncate_dim is not None:
            embeddings = embeddings[:, : self.truncate_dim]
        embeddings = l2_normalize(embeddings)
        return EncodeOutput(
            embeddings=embeddings,
            token_counts=counts,
            truncated=counts > self.max_length,
        )

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": "hf",
            "prompt_prefix": self.prompt_prefix,
            "pooling": self.pooling,
            "normalized_by_model": False,
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
