"""Raw HuggingFace Transformers backend with mean- or CLS-pooling.

Useful for encoders not shipped as SentenceTransformers models (e.g. SimCSE,
plain BERT/RoBERTa). Mean pooling with attention masking is the default.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..config import get_logger, resolve_device
from .base import TextEncoder

_log = get_logger()


class HFMeanPoolEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        batch_size: int = 64,
        normalize: bool = True,
        prompt_prefix: str = "",
        max_seq_length: Optional[int] = None,
        pooling: str = "mean",
        show_progress: bool = True,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {pooling!r}")

        self.name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.normalize = normalize
        self.prompt_prefix = prompt_prefix
        self.max_seq_length = max_seq_length
        self.pooling = pooling
        self.show_progress = show_progress
        self._torch = torch

        _log.info("Loading HF model %s on %s (pooling=%s)", model_name, self.device, pooling)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    def _pool(self, last_hidden, attention_mask):
        torch = self._torch
        if self.pooling == "cls":
            return last_hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        summed = (last_hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        texts = [self.prompt_prefix + t for t in texts] if self.prompt_prefix else list(texts)

        out: List[np.ndarray] = []
        rng = range(0, len(texts), self.batch_size)
        try:
            from tqdm import tqdm

            rng = tqdm(rng, disable=not self.show_progress, desc="encode")
        except ImportError:
            pass

        with torch.no_grad():
            for start in rng:
                batch = texts[start : start + self.batch_size]
                enc = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_seq_length or 512,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**enc).last_hidden_state
                pooled = self._pool(hidden, enc["attention_mask"])
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                out.append(pooled.cpu().numpy().astype(np.float32))
        return np.concatenate(out, axis=0)
