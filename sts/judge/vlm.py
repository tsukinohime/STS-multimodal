"""Vision-language judge: the LLM judge shown the images generated from the sentences.

Scoring is exactly :class:`~sts.judge.scorer.LLMJudge`'s — one forward pass,
the probability-weighted expectation over the answer digits, nothing generated
— but the checkpoint is loaded as an image-text-to-text model with its
processor, and every prompt carries two images: the ones the generator made
from the pair's two sentences, read from the image store. The prompt set
decides what else the prompt holds: nothing of the sentences (``image-v1``,
condition D1) or the sentence pair itself (``text-image-v1``, D2).

For Qwen3.5 the loading class matters. ``AutoModelForCausalLM`` resolves the
checkpoint to ``Qwen3_5MoeForCausalLM``, the language model alone — that is how
the text judge (B0/B1) runs, with the vision weights skipped — while
``AutoModelForImageTextToText`` resolves the same checkpoint to
``Qwen3_5MoeForConditionalGeneration`` with the vision encoder attached. The
language-model weights are the same tensors either way.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from ..config import JudgeSpec, get_logger
from ..images.store import ImageStore, iter_prefetched, visual_tokens
from .prompts import PromptSpec, input_of
from .scorer import LLMJudge, ScoreBatch, model_load_kwargs

_log = get_logger()


@dataclass(frozen=True)
class ImagePrompt:
    """A rendered image-pair prompt: chat text with image slots, and what fills them."""

    text: str                  # the chat template's render, with image placeholders
    images: Tuple[str, ...]    # sentences whose generated images fill the slots, in order


class VLMJudge(LLMJudge):
    def __init__(
        self,
        spec: JudgeSpec,
        device: str,
        dtype: str,
        batch_size: int,
        store: ImageStore,
        progress: bool = True,
    ) -> None:
        self.store = store
        self.input = spec.input
        self.image_size = spec.image_size
        super().__init__(spec, device, dtype, batch_size, progress)

    def _load(self, spec: JudgeSpec, device: str, dtype: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(spec.model_id, trust_remote_code=spec.trust_remote_code)
        model = AutoModelForImageTextToText.from_pretrained(
            spec.model_id, **model_load_kwargs(spec, device, dtype)
        )
        if device != "cuda":
            model = model.to(device)
        tokenizer = self.processor.tokenizer
        template = getattr(self.processor, "chat_template", None) or tokenizer.chat_template or ""
        _log.info("Vision-language judge class: %s", type(model).__name__)
        return tokenizer, model, template

    # -- prompt plumbing ---------------------------------------------------
    def render(self, spec: PromptSpec, sentence_a: str, sentence_b: str) -> ImagePrompt:
        if input_of(spec) != self.input:
            raise ValueError(
                f"{self.key} takes {self.input} input, but prompt {spec.family}/{spec.version} is for "
                f"{input_of(spec)} input; use image-v1 for image, text-image-v1 for text+image"
            )
        kwargs = {}
        if self.template_supports_thinking:
            kwargs["enable_thinking"] = self.enable_thinking
        text = self.processor.apply_chat_template(
            spec.messages(sentence_a, sentence_b),
            tokenize=False,
            add_generation_prompt=True,
            **kwargs,
        )
        return ImagePrompt(text=text, images=(sentence_a, sentence_b))

    # -- scoring -----------------------------------------------------------
    def _load_images(self, batch: Sequence[ImagePrompt]) -> List:
        return [self.store.load(key, self.image_size) for prompt in batch for key in prompt.images]

    def score(self, prompts: Sequence[ImagePrompt], scale: Sequence[int]) -> ScoreBatch:
        torch = self._torch
        prompts = list(prompts)
        self.store.require([key for prompt in prompts for key in prompt.images], purpose=self.key)
        ids = self.digit_token_ids(scale)
        model_dtype = self.model.dtype
        probs_out: List[np.ndarray] = []
        space_out: List[np.ndarray] = []
        len_out: List[np.ndarray] = []

        batches = self._batches(prompts)
        # Images for the next batch are decoded while the GPU scores this one.
        loaded = iter_prefetched(batches, self._load_images)
        with torch.inference_mode():
            for batch, images in self._with_progress(zip(batches, loaded), len(batches)):
                # Images are consumed in placeholder order across the batch:
                # prompt 0's (a, b), then prompt 1's (a, b), and so on.
                encoded = self.processor(
                    text=[prompt.text for prompt in batch],
                    images=images,
                    padding=True,
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                lengths = encoded["attention_mask"].sum(dim=1)
                self._check_lengths(lengths)
                inputs = {key: value.to(self.device) for key, value in encoded.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
                # One prefill, no generation: no KV cache to allocate.
                logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]
                probs, space = self._read_answer(logits, ids)
                probs_out.append(probs)
                space_out.append(space)
                len_out.append(lengths.cpu().numpy())

        return self._collect(probs_out, space_out, len_out)

    def describe(self) -> dict:
        generated = self.store.fingerprint
        return {
            **super().describe(),
            "backend": "hf-vlm",
            "model_class": type(self.model).__name__,
            "input": self.input,
            "image_size": self.image_size,
            "visual_tokens_per_image": visual_tokens(generated["width"], generated["height"], self.image_size),
            **self.store.describe(),
        }
