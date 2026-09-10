"""LLM-as-judge STS: prompt a frozen chat model, take the expected digit.

``prompts``   per-dataset prompt specs with anchors copied from the original
              annotation guidelines, hashed for traceability
``scorer``    one forward pass per pair -> probabilities over the digit tokens
``pipeline``  orchestration mirroring :mod:`sts.pipeline`, reusing its data
              adapters, cache, metrics, manifest and report layers
"""

from .prompts import PROMPT_SETS, PromptSpec, family_of, get_prompt
from .scorer import LLMJudge, ScoreBatch

__all__ = ["PROMPT_SETS", "PromptSpec", "family_of", "get_prompt", "LLMJudge", "ScoreBatch"]
