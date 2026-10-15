"""LLM-as-judge STS: prompt a frozen chat model, take the expected digit.

``prompts``   per-dataset prompt specs with anchors copied from the original
              annotation guidelines, hashed for traceability; text (``v1``),
              image-only (``image-v1``) and text + image (``text-image-v1``) sets
``scorer``    one forward pass per pair -> probabilities over the digit tokens
``vlm``       the same scoring with the pair's two generated images in the prompt
``pipeline``  orchestration mirroring :mod:`sts.pipeline`, reusing its data
              adapters, cache, metrics, manifest and report layers
"""

from .prompts import (
    PROMPT_SETS,
    ImagePairPromptSpec,
    PromptSpec,
    TextImagePairPromptSpec,
    family_of,
    get_prompt,
    input_of,
)
from .scorer import LLMJudge, ScoreBatch
from .vlm import ImagePrompt, VLMJudge

__all__ = [
    "PROMPT_SETS",
    "ImagePairPromptSpec",
    "PromptSpec",
    "TextImagePairPromptSpec",
    "family_of",
    "get_prompt",
    "input_of",
    "LLMJudge",
    "ScoreBatch",
    "ImagePrompt",
    "VLMJudge",
]
