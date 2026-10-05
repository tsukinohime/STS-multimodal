"""LLM judge: one forward pass per pair, expectation over the digit tokens.

No text is generated. The chat template is rendered up to the point where the
assistant's first token would appear (with thinking disabled, so no reasoning
block precedes it), and the model's next-token distribution at that position is
read off for the allowed digit tokens. The prediction is

    E[score] = sum_d d * p(d) / sum_d p(d)

which turns a 6-way (or 5-, 7-way) categorical answer into a continuous score
and thereby removes the massive rank ties an argmax answer would produce. The
argmax is kept alongside for comparison, as is the total probability mass the
model put on valid digits — a low mass means the model wanted to say something
else, and those rows are flagged rather than silently trusted.

Batched with left padding so the last position of every row is the answer slot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..config import JudgeSpec, get_logger, torch_dtype
from ..hashing import sha256_text
from .prompts import PromptSpec

_log = get_logger()


@dataclass
class ScoreBatch:
    raw_probs: np.ndarray     # (n, k) p(digit) before renormalisation
    space_mass: np.ndarray    # (n,)  p(" "): a leading space means the digit was not next
    prompt_tokens: np.ndarray # (n,)  rendered prompt length in tokens

    @property
    def valid_mass(self) -> np.ndarray:
        return self.raw_probs.sum(axis=1)

    def normalised(self, min_valid_mass: float) -> np.ndarray:
        mass = self.valid_mass
        with np.errstate(divide="ignore", invalid="ignore"):
            norm = self.raw_probs / mass[:, None]
        norm[mass < min_valid_mass] = np.nan
        return norm

    def expected(self, scale: Sequence[int], min_valid_mass: float) -> np.ndarray:
        return (self.normalised(min_valid_mass) * np.asarray(scale, dtype=np.float64)).sum(axis=1)

    def argmax(self, scale: Sequence[int], min_valid_mass: float) -> np.ndarray:
        norm = self.normalised(min_valid_mass)
        values = np.asarray(scale, dtype=np.float64)
        out = values[np.nanargmax(np.nan_to_num(norm, nan=-1.0), axis=1)]
        out[np.isnan(norm).any(axis=1)] = np.nan
        return out


def model_load_kwargs(spec: JudgeSpec, device: str, dtype: str) -> dict:
    kwargs = {"dtype": torch_dtype(dtype), "trust_remote_code": spec.trust_remote_code}
    if device == "cuda":
        # Stream straight onto the GPU; a ~70 GB checkpoint must not be
        # staged through host RAM twice.
        kwargs["device_map"] = "cuda"
    return kwargs


class LLMJudge:
    """A frozen chat LLM used as a pairwise similarity rater."""

    def __init__(
        self,
        spec: JudgeSpec,
        device: str,
        dtype: str,
        batch_size: int,
        progress: bool = True,
    ) -> None:
        import torch

        self.key = spec.key
        self.model_id = spec.model_id
        self.device = device
        self.dtype_name = dtype
        self.batch_size = batch_size
        self.progress = progress
        self.enable_thinking = spec.enable_thinking
        self.max_prompt_tokens = spec.max_prompt_tokens
        self._torch = torch

        _log.info("Loading judge %s on %s / %s", spec.model_id, device, dtype)
        self.tokenizer, self.model, template = self._load(spec, device, dtype)
        # Left padding: the answer slot must be the last position of every row.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

        self.template_hash = sha256_text(template)
        self.template_supports_thinking = "enable_thinking" in template
        if self.template_supports_thinking:
            _log.info("Chat template has a thinking switch; enable_thinking=%s", self.enable_thinking)
        else:
            _log.info("Chat template has no thinking switch (fine: the answer is the first token)")

        space_ids = self.tokenizer.encode(" ", add_special_tokens=False)
        self._space_id: Optional[int] = space_ids[0] if len(space_ids) == 1 else None
        self._digit_cache: dict = {}

    def _load(self, spec: JudgeSpec, device: str, dtype: str):
        """``(tokenizer, model, chat_template)``; :class:`~sts.judge.vlm.VLMJudge` overrides it."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(spec.model_id, **model_load_kwargs(spec, device, dtype))
        if device != "cuda":
            model = model.to(device)
        return tokenizer, model, tokenizer.chat_template or ""

    # -- prompt plumbing ---------------------------------------------------
    def digit_token_ids(self, scale: Sequence[int]) -> List[int]:
        """Token id of each allowed answer. Every digit must be exactly one token."""
        key = tuple(scale)
        if key not in self._digit_cache:
            ids = []
            for value in scale:
                enc = self.tokenizer.encode(str(value), add_special_tokens=False)
                if len(enc) != 1:
                    raise ValueError(
                        f"{self.model_id}: answer {value!r} tokenises to {enc}, not a single token; "
                        "expectation over next-token probabilities needs single-token answers"
                    )
                ids.append(enc[0])
            self._digit_cache[key] = ids
        return self._digit_cache[key]

    def render(self, spec: PromptSpec, sentence_a: str, sentence_b: str) -> str:
        kwargs = {}
        if self.template_supports_thinking:
            kwargs["enable_thinking"] = self.enable_thinking
        return self.tokenizer.apply_chat_template(
            spec.messages(sentence_a, sentence_b),
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )

    def prompt_hash(self, spec: PromptSpec) -> str:
        """Spec hash combined with the chat template, i.e. the bytes the model saw."""
        return sha256_text(f"{spec.spec_hash}|{self.template_hash}|thinking={self.enable_thinking}")

    # -- scoring -----------------------------------------------------------
    def _batches(self, prompts: Sequence) -> List[Sequence]:
        return [prompts[i : i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]

    def _with_progress(self, iterable, total: int):
        if self.progress:
            try:
                from tqdm import tqdm

                return tqdm(iterable, total=total, desc=f"judge[{self.key}]", unit="batch", leave=False)
            except ImportError:
                pass
        return iterable

    def _check_lengths(self, lengths) -> None:
        too_long = int((lengths > self.max_prompt_tokens).sum())
        if too_long:
            _log.warning("%d prompt(s) exceed max_prompt_tokens=%d (not truncated)",
                         too_long, self.max_prompt_tokens)

    def _read_answer(self, logits, ids: List[int]):
        """``(p(digit) per allowed digit, p(" "))`` at the answer position, per row."""
        torch = self._torch
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        probs = logprobs[:, ids].exp().cpu().numpy()
        if self._space_id is not None:
            space = logprobs[:, self._space_id].exp().cpu().numpy()
        else:
            space = np.zeros(logits.shape[0], dtype=np.float32)
        return probs, space

    @staticmethod
    def _collect(probs_out, space_out, len_out) -> ScoreBatch:
        return ScoreBatch(
            raw_probs=np.concatenate(probs_out, axis=0).astype(np.float64),
            space_mass=np.concatenate(space_out, axis=0).astype(np.float64),
            prompt_tokens=np.concatenate(len_out, axis=0).astype(np.int32),
        )

    def score(self, prompts: Sequence[str], scale: Sequence[int]) -> ScoreBatch:
        torch = self._torch
        prompts = list(prompts)
        ids = self.digit_token_ids(scale)
        probs_out: List[np.ndarray] = []
        space_out: List[np.ndarray] = []
        len_out: List[np.ndarray] = []

        batches = self._batches(prompts)
        with torch.inference_mode():
            for batch in self._with_progress(batches, len(batches)):
                # The template already contains every special token, so the
                # tokenizer must not add another BOS.
                encoded = self.tokenizer(
                    batch, return_tensors="pt", padding=True, add_special_tokens=False
                ).to(self.device)
                lengths = encoded["attention_mask"].sum(dim=1)
                self._check_lengths(lengths)
                try:
                    # Only the last position is needed; materialising logits
                    # for every position over a 250k vocab would cost GBs.
                    logits = self.model(**encoded, logits_to_keep=1).logits[:, -1, :]
                except TypeError:
                    logits = self.model(**encoded).logits[:, -1, :]
                probs, space = self._read_answer(logits, ids)
                probs_out.append(probs)
                space_out.append(space)
                len_out.append(lengths.cpu().numpy())

        return self._collect(probs_out, space_out, len_out)

    def describe(self) -> dict:
        return {
            "model_id": self.model_id,
            "backend": "hf",
            "task": "llm-judge",
            "prompt_prefix": "",
            "pooling": "next-token expectation over digit tokens",
            "dtype": self.dtype_name,
            "device": self.device,
            "chat_template_sha256": self.template_hash,
            "enable_thinking": self.enable_thinking,
            "padding_side": "left",
            "max_prompt_tokens": self.max_prompt_tokens,
        }

    def close(self) -> None:
        if getattr(self, "model", None) is None:
            return
        import gc

        del self.model
        self.model = None
        gc.collect()
        torch = self._torch
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
