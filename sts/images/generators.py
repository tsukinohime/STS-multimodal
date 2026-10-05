"""Text-to-image generators: one sentence in, one image out.

``flux2-klein``  FLUX.2 [klein] 4B through diffusers' ``Flux2KleinPipeline``: the
                 generator of the image conditions. Step-distilled, so 4 steps
                 and no classifier-free guidance; prompts are encoded by its
                 Qwen3-4B text encoder through the chat template.
``render-text``  a stand-in that typesets the sentence on a white canvas. No
                 model, no download, deterministic — it exercises the image
                 pipeline end to end on a laptop. A VLM simply reads the text
                 back, so numbers produced with it say nothing about generation.

Seeds. ``seed_policy: shared`` starts every sentence from the same noise
(``seed``); ``per-text`` derives the seed from the sentence. Either way an image
is a function of its sentence alone, never of batch order or size: each prompt
gets its own CPU ``torch.Generator``, which also makes the starting noise
identical on a laptop and on a cluster GPU. Shared noise is the default, so two
sentences' images differ only through what their texts change, not also
through an unrelated random layout.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Sequence

import numpy as np

from ..config import GeneratorSpec, get_logger, torch_dtype
from ..hashing import sha256_text

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

_log = get_logger()

SEED_POLICIES = ("shared", "per-text")


def seed_for(text: str, spec: GeneratorSpec) -> int:
    """The generation seed of ``text`` under ``spec.seed_policy``."""
    if spec.seed_policy == "shared":
        return int(spec.seed)
    if spec.seed_policy == "per-text":
        return int(sha256_text(f"{spec.seed}\x1f{text}")[:8], 16)
    raise ValueError(f"seed_policy must be one of {SEED_POLICIES}, got {spec.seed_policy!r}")


class ImageGenerator(ABC):
    """A frozen text-to-image model."""

    def __init__(self, spec: GeneratorSpec) -> None:
        self.spec = spec
        self.key = spec.key

    @abstractmethod
    def generate(self, texts: Sequence[str], seeds: Sequence[int]) -> List["Image"]:
        """One RGB image per text, in order."""

    def prompt_tokens(self, texts: Sequence[str]) -> np.ndarray:
        """Prompt length in the generator's own tokens (0 when it has no tokenizer)."""
        return np.zeros(len(texts), dtype=np.int32)

    def describe(self) -> Dict[str, object]:
        return {"backend": self.spec.backend, "model_id": self.spec.model_id, **self.spec.fingerprint()}

    def close(self) -> None:
        """Release device memory. Safe to call twice."""


class Flux2KleinGenerator(ImageGenerator):
    def __init__(self, spec: GeneratorSpec, device: str) -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        super().__init__(spec)
        self.device = device
        self._torch = torch

        _log.info("Loading generator %s on %s / %s", spec.model_id, device, spec.dtype)
        self.pipe = Flux2KleinPipeline.from_pretrained(spec.model_id, dtype=torch_dtype(spec.dtype))
        self.pipe.to(device)
        # One progress bar per call would print a line per batch; the caller
        # keeps one bar for the whole run instead.
        self.pipe.set_progress_bar_config(disable=True)
        # The VAE decode of a 1024 px batch is the memory peak; decode one at a time.
        self.pipe.vae.enable_slicing()

        self.distilled = bool(self.pipe.config.get("is_distilled", False))
        if self.distilled and spec.guidance_scale > 1.0:
            _log.warning("%s is step-distilled: guidance_scale=%s is ignored by the pipeline",
                         spec.model_id, spec.guidance_scale)
        if not self.distilled and spec.num_inference_steps < 20:
            _log.warning("%s is not step-distilled, yet num_inference_steps=%d; the base "
                         "checkpoints expect ~50", spec.model_id, spec.num_inference_steps)

    def prompt_tokens(self, texts: Sequence[str]) -> np.ndarray:
        """Length of each prompt as the pipeline encodes it (chat template, thinking off)."""
        tokenizer = self.pipe.tokenizer
        rendered = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for text in texts
        ]
        ids = tokenizer(rendered, padding=False, truncation=False)["input_ids"]
        return np.array([len(row) for row in ids], dtype=np.int32)

    def generate(self, texts: Sequence[str], seeds: Sequence[int]) -> List["Image"]:
        generators = [self._torch.Generator("cpu").manual_seed(int(seed)) for seed in seeds]
        output = self.pipe(
            prompt=list(texts),
            height=self.spec.height,
            width=self.spec.width,
            num_inference_steps=self.spec.num_inference_steps,
            guidance_scale=self.spec.guidance_scale,
            generator=generators,
            max_sequence_length=self.spec.max_sequence_length,
            output_type="pil",
        )
        return list(output.images)

    def describe(self) -> Dict[str, object]:
        import diffusers

        return {
            **super().describe(),
            "pipeline": type(self.pipe).__name__,
            "is_distilled": self.distilled,
            "device": self.device,
            "diffusers": diffusers.__version__,
        }

    def close(self) -> None:
        if getattr(self, "pipe", None) is None:
            return
        import gc

        del self.pipe
        self.pipe = None
        gc.collect()
        torch = self._torch
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()


class RenderTextGenerator(ImageGenerator):
    """Typesets the sentence: a model-free stand-in for exercising the pipeline."""

    def generate(self, texts: Sequence[str], seeds: Sequence[int]) -> List["Image"]:
        from PIL import Image, ImageDraw

        width, height = self.spec.width, self.spec.height
        font = font_of_size(max(12, height // 16))
        line_height = int(getattr(font, "size", 11) * 1.3)
        margin = width // 16
        images = []
        for text in texts:
            canvas = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(canvas)
            for row, line in enumerate(wrap_text(draw, text, font, width - 2 * margin)):
                draw.text((margin, margin + row * line_height), line, fill="black", font=font)
            images.append(canvas)
        return images


def font_of_size(size: int):
    """Pillow's bundled font at ``size`` px; the fixed bitmap font on Pillow < 10.1."""
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def wrap_text(draw, text: str, font, max_width: int) -> List[str]:
    """Greedy word wrap by rendered width; a single over-long word overflows."""
    lines: List[str] = []
    for word in text.split():
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and draw.textlength(candidate, font=font) <= max_width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def build_generator(spec: GeneratorSpec, device: str) -> ImageGenerator:
    """Instantiate the generator named by ``spec.backend``."""
    if spec.seed_policy not in SEED_POLICIES:
        raise ValueError(f"seed_policy must be one of {SEED_POLICIES}, got {spec.seed_policy!r}")
    backend = spec.backend.strip().lower()
    if backend in ("flux2-klein", "flux2_klein"):
        return Flux2KleinGenerator(spec, device)
    if backend in ("render-text", "render_text"):
        return RenderTextGenerator(spec)
    raise ValueError(f"unknown generator backend {spec.backend!r}; use flux2-klein or render-text")
