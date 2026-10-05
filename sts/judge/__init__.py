"""LLM-as-judge STS: prompt a frozen chat model, take the expected digit.

``prompts``   per-dataset prompt specs with anchors copied from the original
              annotation guidelines, hashed for traceability; text (``v1``) and
              image-only (``image-v1``) sets
``scorer``    one forward pass per pair -> probabilities over the digit tokens
``vlm``       the same scoring with the pair shown as two generated images
``pipeline``  orchestration mirroring :mod:`sts.pipeline`, reusing its data
              adapters, cache, metrics, manifest and report layers
"""

from .prompts import PROMPT_SETS, ImagePairPromptSpec, PromptSpec, family_of, get_prompt, input_of
from .scorer import LLMJudge, ScoreBatch
from .vlm import ImagePrompt, VLMJudge

__all__ = [
    "PROMPT_SETS",
    "ImagePairPromptSpec",
    "PromptSpec",
    "family_of",
    "get_prompt",
    "input_of",
    "LLMJudge",
    "ScoreBatch",
    "ImagePrompt",
    "VLMJudge",
]
