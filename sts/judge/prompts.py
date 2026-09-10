"""Prompt specifications for LLM-as-judge STS, one per dataset family.

Design rules (from the experiment brief):

* **Scale and anchors are copied from each dataset's original annotation
  guidelines**, never invented. Every :class:`PromptSpec` records its source.
  Where the original authors deliberately gave annotators *no* verbal
  definitions (SICK, STS3k), the prompt gives none either — adding some would
  measure a different construct than the human gold scores.
* **Symmetric wording.** The question is always about "the two sentences";
  neither sentence is the subject or object of the comparison.
* **No chain-of-thought.** The answer is a single digit and nothing else.
* **Hashed.** ``spec_hash`` covers every character of the template so any later
  edit produces a new hash; the run manifest and every prediction row carry it.

Scales differ across datasets (CxC 0–5, SICK 1–5, STS3k 1–7) because that is
what the guidelines define. Spearman is invariant to the scale, so this costs
nothing for the comparison across datasets. Changing a scale means writing a
new spec version, which is the point: the hash changes with it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, Tuple

from ..hashing import sha256_text

# --------------------------------------------------------------------------- #
# Shared wording
# --------------------------------------------------------------------------- #
_SYSTEM = (
    "You are a careful annotator rating the semantic similarity of sentence pairs. "
    "You always answer with a single digit and nothing else."
)

_PAIR_BLOCK = "Sentence pair:\n- {sentence_a}\n- {sentence_b}"


def _answer_line(scale: Tuple[int, ...]) -> str:
    digits = ", ".join(str(v) for v in scale[:-1]) + f", or {scale[-1]}"
    return (
        f"Reply with exactly one digit ({digits}). "
        "Output nothing else: no words, no punctuation, no explanation."
    )


@dataclass(frozen=True)
class PromptSpec:
    """One dataset family's judging prompt, fully specified and hashable."""

    family: str                       # cxc | sick | sts3k
    version: str                      # bump when any wording changes
    scale: Tuple[int, ...]            # the discrete answers the judge may give
    system: str
    instruction: str                  # the task + scale text, verbatim anchors inside
    source: str                       # where the anchors were copied from
    notes: str = ""

    def render_user(self, sentence_a: str, sentence_b: str) -> str:
        return "\n\n".join([
            self.instruction,
            _PAIR_BLOCK.format(sentence_a=sentence_a, sentence_b=sentence_b),
            _answer_line(self.scale),
        ])

    def messages(self, sentence_a: str, sentence_b: str) -> list:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.render_user(sentence_a, sentence_b)},
        ]

    @property
    def spec_hash(self) -> str:
        """Hash of everything that reaches the model, minus the chat template.

        The pair block and answer line are part of the render, so they are
        included by hashing a render with fixed placeholder sentences.
        """
        payload = {
            **asdict(self),
            "rendered_with_placeholders": self.render_user("{sentence_a}", "{sentence_b}"),
        }
        return sha256_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    def to_dict(self) -> dict:
        return {**asdict(self), "spec_hash": self.spec_hash}


# --------------------------------------------------------------------------- #
# v1 — anchors verbatim from the original guidelines
# --------------------------------------------------------------------------- #

# CxC caption–caption STS. The CxC paper (Parekh et al., 2021, appendix) states
# raters used the STS definitions of Cer et al. (2017) / Agirre et al. (2012),
# reproduced there as follows. Raters could give non-integer scores (e.g. 1.3);
# gold = mean of five raters.
_CXC_V1 = PromptSpec(
    family="cxc",
    version="v1",
    scale=(0, 1, 2, 3, 4, 5),
    system=_SYSTEM,
    instruction=(
        "Rate how similar in meaning the two sentences are, on a scale from 0 to 5, "
        "using these definitions:\n"
        "5: The texts are completely equivalent as they mean the same thing.\n"
        "4: The texts are mostly equivalent but some unimportant details differ.\n"
        "3: The texts are roughly equivalent but some important information differs or is missing.\n"
        "2: The texts are not equivalent but share some details.\n"
        "1: The texts are not equivalent but are on the same topic.\n"
        "0: The texts are on different topics."
    ),
    source=(
        "Parekh et al. 2021, 'Crisscrossed Captions' (arXiv:2004.15020), appendix table of "
        "STS rating definitions shown to annotators; these are the STS-Benchmark definitions "
        "of Cer et al. 2017 / Agirre et al. 2012. Copied verbatim."
    ),
    notes="Original raters could answer with non-integers (e.g. 1.3); gold is the mean of 5 ratings.",
)

# SICK relatedness. Marelli et al. 2014 (LREC, §3.4): "the degree to which two
# sentence meanings are related (on a 5-point scale)". Crucially: "In order to
# clarify the task to non-expert participants, while avoiding biasing their
# judgments with strict definitions, the instructions described the task only
# through examples", giving exactly one example pair for each endpoint. The
# prompt therefore contains those two examples and no level definitions.
_SICK_V1 = PromptSpec(
    family="sick",
    version="v1",
    scale=(1, 2, 3, 4, 5),
    system=_SYSTEM,
    instruction=(
        "Rate the degree to which the meanings of the two sentences are related, "
        "on a 5-point scale from 1 to 5.\n"
        "The scale is described only by these two examples:\n"
        "- The two sentences are completely unrelated (1):\n"
        "    Two girls are playing outdoors near a woman\n"
        "    The elephant is being ridden by the man\n"
        "- The two sentences are very related (5):\n"
        "    A man is cooking pancakes\n"
        "    The man is cooking pancakes"
    ),
    source=(
        "Marelli et al. 2014, 'A SICK cure for the evaluation of compositional distributional "
        "semantic models' (LREC), §3.4: task = 'the degree to which two sentence meanings are "
        "related (on a 5-point scale)'; the two endpoint examples ('A and B are completely "
        "unrelated' / 'A and B are very related') are copied verbatim. The original gives no "
        "per-level definitions by design. Mapping the endpoints to 1 and 5 follows from "
        "'5-point scale'. Gold = mean of 10 ratings, 5 per presentation order."
    ),
    notes="Original wording labels the sentences 'A' and 'B'; rendered here as 'the two sentences'.",
)

# STS3k. Fodor, De Deyne & Suzuki 2025 (Computational Linguistics 51(1)),
# appendix 8.2.1 "Instructions to Participants", rating portion verbatim. The
# authors explicitly gave "no special training or instructions about how to
# allocate ratings". 7-point Likert; released scores are the mean normalised
# to [0, 1].
_STS3K_V1 = PromptSpec(
    family="sts3k",
    version="v1",
    scale=(1, 2, 3, 4, 5, 6, 7),
    system=_SYSTEM,
    instruction=(
        "Your task is to judge how similar is the meaning of the two sentences. "
        "You will make this judgement by choosing a rating from 1 (very dissimilar) to 7 (very similar). "
        "In providing your rating, consider both the similarity in meaning of the individual words "
        "contained in the sentences, as well as the similarity of the overall idea or meaning expressed "
        "by the sentences. Some of the sentences may be slightly unusual or ambiguous; nevertheless you "
        "should do your best to understand their likely meaning. Bear in mind that we are not looking "
        "for any one specific 'right answer' or strategy in your responses. Your task is simply to make "
        "a judgement about how similar you think is the meaning of the two paired sentences. The only "
        "exception is that if you find a sentence that truly does not make any sense at all, then you "
        "should give it a very low similarity to whatever it is paired with. In all other cases, make "
        "your best judgement based on your assessment of overall meaning of the sentences."
    ),
    source=(
        "Fodor, De Deyne & Suzuki 2025, 'Compositionality and Sentence Meaning' (Computational "
        "Linguistics 51(1)), appendix 8.2.1 'Instructions to Participants', rating instructions "
        "copied verbatim; questionnaire logistics (time limit, thanks) omitted. §4.2: 7-point "
        "Likert (1–7), ~20 ratings per pair, mean normalised to [0, 1]."
    ),
    notes="Authors deliberately provided no rating framework; none is added here.",
)

PROMPT_SETS: Dict[str, Dict[str, PromptSpec]] = {
    "v1": {"cxc": _CXC_V1, "sick": _SICK_V1, "sts3k": _STS3K_V1},
}


def family_of(dataset_name: str) -> str:
    """``cxc-val`` -> ``cxc``, ``sick-test`` -> ``sick``, ``sts3k`` -> ``sts3k``."""
    return dataset_name.split("-", 1)[0].lower()


def get_prompt(prompt_set: str, dataset_name: str) -> PromptSpec:
    if prompt_set not in PROMPT_SETS:
        raise KeyError(f"unknown prompt set {prompt_set!r}; have {sorted(PROMPT_SETS)}")
    family = family_of(dataset_name)
    specs = PROMPT_SETS[prompt_set]
    if family not in specs:
        raise KeyError(f"prompt set {prompt_set!r} has no prompt for dataset family {family!r}")
    return specs[family]
