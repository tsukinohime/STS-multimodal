"""jina-embeddings-v5 adapter (``v5-text-small`` and ``v5-omni-small``).

Both checkpoints expose the same official inference entry point via
``trust_remote_code=True``::

    model = AutoModel.from_pretrained(repo, trust_remote_code=True)
    emb = model.encode(texts=[...], task="text-matching", prompt_name="document")

That call is what the model card prescribes, and it owns every detail this
project must not get wrong:

* **prefix**  — ``"Document: "`` for ``prompt_name="document"``, which is the
  only prompt the non-retrieval tasks (text-matching / classification /
  clustering) were trained with; ``"Query: "`` otherwise.
* **adapter** — ``set_adapter([task])`` swaps in the task LoRA plus the
  task-specific special-token embeddings before the forward pass.
* **pooling** — last non-padding token.
* **norm**    — the returned vectors are already L2-normalised, so cosine
  similarity is a plain dot product.

Nothing is reimplemented here; the adapter only adds batching, dtype/device
placement, and truncation accounting.

``modality="text"`` on the omni checkpoint skips loading the vision and audio
towers. Text embeddings are identical either way — the towers are separate
parameters — so the text-only milestone gets the cheap load, and the visual
milestone reloads the same repo with ``modality="vision"`` to obtain image
embeddings in this very same space.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from ..config import ModelSpec, get_logger, torch_dtype
from .base import EncodeOutput, TextEncoder, batched, l2_normalize

_log = get_logger()

_DEFAULT_TASK = "text-matching"
_DEFAULT_PROMPT = "document"
#: The literal prefixes the checkpoints prepend, mirrored here only so token
#: counts (and therefore truncation flags) are measured on the real input.
_PROMPT_PREFIXES = {"query": "Query: ", "document": "Document: "}


class JinaV5Encoder(TextEncoder):
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

        self.key = spec.key
        self.model_id = spec.model_id
        self.task = spec.task or _DEFAULT_TASK
        self.prompt_name = spec.prompt_name or _DEFAULT_PROMPT
        self.truncate_dim = spec.truncate_dim
        self.modality = spec.modality
        self.device = device
        self.dtype_name = dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.progress = progress
        self._torch = torch

        if self.prompt_name not in _PROMPT_PREFIXES:
            raise ValueError(
                f"{spec.key}: prompt_name must be 'document' or 'query', got {self.prompt_name!r}"
            )

        load_kwargs = {"trust_remote_code": spec.trust_remote_code, "dtype": torch_dtype(dtype)}
        if self.modality:
            load_kwargs["modality"] = self.modality

        _log.info(
            "Loading %s (task=%s, prompt=%s, modality=%s) on %s / %s",
            spec.model_id, self.task, self.prompt_name, self.modality or "default", device, dtype,
        )
        self.model = AutoModel.from_pretrained(spec.model_id, **load_kwargs)
        self.model = self.model.to(device).eval()

        # Both checkpoints are PEFT-wrapped, and which object carries the real
        # config differs between them: omni reads base_model.model.config,
        # text reads base_model.config. Check whichever is present so a typo in
        # `task` fails at load rather than on the first batch.
        available = None
        for holder in (getattr(self.model, "config", None),
                       getattr(getattr(self.model, "base_model", None), "config", None),
                       getattr(getattr(getattr(self.model, "base_model", None), "model", None),
                               "config", None)):
            available = getattr(holder, "task_names", None)
            if available:
                break
        if available and self.task not in available:
            raise ValueError(f"{spec.model_id}: task {self.task!r} not in {available}")

        # The omni checkpoint carries its tokenizer on the model; the text
        # checkpoint may not, so fall back to loading it from the same repo.
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                spec.model_id, trust_remote_code=spec.trust_remote_code
            )
        self.tokenizer = tokenizer
        self._dim: Optional[int] = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.encode(["dimension probe"]).embeddings.shape[1])
        return self._dim

    def token_lengths(self, texts: Sequence[str]) -> np.ndarray:
        prefix = _PROMPT_PREFIXES[self.prompt_name]
        encoded = self.tokenizer(
            [f"{prefix}{t}" for t in texts], padding=False, truncation=False
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
        chunks: List[np.ndarray] = []
        batches = batched(texts, self.batch_size)
        iterator = batches
        if self.progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(batches, desc=f"encode[{self.key}]", unit="batch", leave=False)
            except ImportError:
                pass

        # no_grad rather than inference_mode: the checkpoint's encode() swaps
        # task adapters by writing into the embedding weights, and in-place
        # weight mutation under inference_mode is a trap. The model applies its
        # own no_grad internally too; this just makes the outer scope explicit.
        with self._torch.no_grad():
            for batch in iterator:
                vectors = self.model.encode(
                    texts=list(batch),
                    task=self.task,
                    prompt_name=self.prompt_name,
                    truncate_dim=self.truncate_dim,
                    max_length=self.max_length,
                )
                chunks.append(vectors.float().cpu().numpy())

        embeddings = np.concatenate(chunks, axis=0).astype(np.float32)
        # The checkpoints normalise internally; re-normalising is a cheap guard
        # against a dtype round-trip leaving norms at 0.999x.
        embeddings = l2_normalize(embeddings)
        return EncodeOutput(
            embeddings=embeddings,
            token_counts=counts,
            truncated=counts > self.max_length,
        )

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": "jina-v5",
            "task": self.task,
            "prompt_name": self.prompt_name,
            "prompt_prefix": _PROMPT_PREFIXES[self.prompt_name],
            "pooling": "last-token (model-internal)",
            "normalized_by_model": True,
            "modality": self.modality,
            "truncate_dim": self.truncate_dim,
            "max_length": self.max_length,
            "dtype": self.dtype_name,
            "device": self.device,
        }

    def close(self) -> None:
        model = getattr(self, "model", None)
        if model is None:
            return
        del self.model
        self.model = None
        _free_device_memory(self._torch, self.device)


def _free_device_memory(torch, device: str) -> None:
    import gc

    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and getattr(torch.backends, "mps", None) is not None:
        empty = getattr(getattr(torch, "mps", None), "empty_cache", None)
        if empty is not None:
            empty()
