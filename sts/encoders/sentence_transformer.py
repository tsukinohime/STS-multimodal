"""SentenceTransformers (SBERT) backend — the default encoder.

Handles device placement, batching and (optional) L2-normalization internally.
A ``prompt_prefix`` is prepended to every text for instruction-tuned encoders
that expect one (e.g. E5's ``"query: "``); leave empty for MPNet / MiniLM.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from ..config import get_logger, resolve_device
from .base import TextEncoder

_log = get_logger()


class SentenceTransformerEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = True,
        prompt_prefix: str = "",
        max_seq_length: Optional[int] = None,
        show_progress: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.normalize = normalize
        self.prompt_prefix = prompt_prefix
        self.show_progress = show_progress

        _log.info("Loading SentenceTransformer %s on %s", model_name, self.device)
        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if self.prompt_prefix:
            texts = [self.prompt_prefix + t for t in texts]
        emb = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=self.show_progress,
        )
        return emb.astype(np.float32)
