"""Generated-image input for an encoder: sentences in, embeddings of their images out.

Wraps an encoder that can embed images (``encode_images``) behind the plain
:class:`TextEncoder` interface, so the text pipeline runs unchanged — dedup,
embedding cache, cosine, metrics. A "text" handed to :meth:`encode` is only the
key of the image the generator made from it; the sentence itself never reaches
the model.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..images.store import ImageStore, iter_prefetched, visual_tokens
from .base import EncodeOutput, TextEncoder, batched


class GeneratedImageEncoder(TextEncoder):
    def __init__(self, base: TextEncoder, store: ImageStore, image_size: Optional[int] = None) -> None:
        if not hasattr(base, "encode_images"):
            raise ValueError(f"{base.key}: this backend cannot embed images (input: image)")
        self.base = base
        self.store = store
        self.image_size = image_size
        self.key = base.key
        self.model_id = base.model_id
        # The pipeline flags a text as truncated when its token count exceeds
        # max_length. Images are resized, never cut, and their token counts are
        # image tokens, so the text limit must not apply to them.
        self.max_length = np.iinfo(np.int32).max
        self.batch_size = getattr(base, "batch_size", 16)
        self.progress = getattr(base, "progress", False)

    def token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        """Image tokens per sentence's image (a constant when the generator makes one size)."""
        counts = []
        for text in texts:
            entry = self.store.entry(text) or {}
            width, height = entry.get("width"), entry.get("height")
            counts.append(visual_tokens(width, height, self.image_size) if width and height else 0)
        return np.array(counts, dtype=np.int32)

    def _load(self, texts: Sequence[str]) -> List:
        return [self.store.load(text, self.image_size) for text in texts]

    def encode(self, texts: Sequence[str]) -> EncodeOutput:
        texts = list(texts)
        self.store.require(texts, purpose=self.key)
        batches = batched(texts, self.batch_size)
        iterator = iter_prefetched(batches, self._load)
        if self.progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=len(batches), desc=f"encode-images[{self.key}]",
                                unit="batch", leave=False)
            except ImportError:
                pass
        chunks = [self.base.encode_images(images) for images in iterator]
        embeddings = (np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0))).astype(np.float32)
        return EncodeOutput(
            embeddings=embeddings,
            token_counts=self.token_lengths(texts),
            # The processors resize images instead of cutting them: nothing is truncated.
            truncated=np.zeros(len(texts), dtype=bool),
        )

    @property
    def dim(self) -> int:
        return self.base.dim

    def describe(self) -> dict:
        return {
            **self.base.describe(),
            "input": "image",
            "image_size": self.image_size,
            **self.store.describe(),
        }

    def close(self) -> None:
        self.base.close()
