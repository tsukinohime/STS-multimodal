"""Unified schema every dataset adapter emits.

Each adapter turns its native on-disk format into a list of :class:`STSPair`
with exactly the same fields, so everything downstream never needs to know
which corpus it is looking at. ``image1_id`` / ``image2_id`` are empty strings
for text-only corpora; CxC fills them with COCO image ids so the same pair
table can be reused for the later generated-image experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: Column order used for every on-disk pair table.
PAIR_COLUMNS: Tuple[str, ...] = (
    "pair_id",
    "dataset",
    "split",
    "domain",
    "sentence1_id",
    "sentence2_id",
    "sentence1",
    "sentence2",
    "gold_score",
    "image1_id",
    "image2_id",
)


def clean_text(text: object) -> str:
    """The only normalisation applied to any sentence.

    Strips leading/trailing whitespace (COCO captions routinely carry a
    trailing space and a newline). Case, punctuation and stop words are left
    untouched on purpose — encoders are evaluated on the text as distributed.
    """
    return str(text).strip()


@dataclass(frozen=True)
class STSPair:
    pair_id: str
    dataset: str
    split: str
    domain: str
    sentence1_id: str
    sentence2_id: str
    sentence1: str
    sentence2: str
    gold_score: float
    image1_id: str = ""
    image2_id: str = ""


@dataclass
class STSDataset:
    """A loaded corpus plus the provenance needed for the run manifest."""

    name: str
    split: str
    pairs: List[STSPair]
    score_range: Optional[Tuple[float, float]] = None
    source_files: Dict[str, str] = field(default_factory=dict)  # path -> sha256
    #: Rows present in the source file(s) before any filtering, for the
    #: "actual pair count matches the data file" validation check.
    n_rows_in_source: Optional[int] = None
    notes: Dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.pairs)

    @property
    def gold(self) -> np.ndarray:
        return np.array([p.gold_score for p in self.pairs], dtype=np.float64)

    @property
    def sentence1(self) -> List[str]:
        return [p.sentence1 for p in self.pairs]

    @property
    def sentence2(self) -> List[str]:
        return [p.sentence2 for p in self.pairs]

    @property
    def domains(self) -> List[str]:
        return [p.domain for p in self.pairs]

    def unique_texts(self) -> List[str]:
        """Deduplicated texts in first-seen order (CxC reuses captions heavily)."""
        return list(dict.fromkeys(self.sentence1 + self.sentence2))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(p) for p in self.pairs], columns=list(PAIR_COLUMNS))

    def subsample(self, limit: Optional[int], seed: int = 0) -> "STSDataset":
        """Deterministic subset used by the smoke config.

        Sampling is stratified over ``domain`` when the dataset has more than
        one, so a 32-pair smoke test still exercises every category.
        """
        if limit is None or limit >= len(self):
            return self
        rng = np.random.default_rng(seed)
        domains = np.array(self.domains)
        unique_domains = list(dict.fromkeys(domains.tolist()))

        if len(unique_domains) > 1:
            picked: List[int] = []
            per_domain = max(1, limit // len(unique_domains))
            for dom in unique_domains:
                idx = np.flatnonzero(domains == dom)
                take = min(per_domain, len(idx))
                picked.extend(rng.choice(idx, size=take, replace=False).tolist())
            remaining = limit - len(picked)
            if remaining > 0:
                pool = np.setdiff1d(np.arange(len(self)), np.array(picked, dtype=int))
                if len(pool):
                    picked.extend(
                        rng.choice(pool, size=min(remaining, len(pool)), replace=False).tolist()
                    )
            index: Sequence[int] = sorted(picked)[:limit]
        else:
            index = sorted(rng.choice(len(self), size=limit, replace=False).tolist())

        return STSDataset(
            name=self.name,
            split=self.split,
            pairs=[self.pairs[i] for i in index],
            score_range=self.score_range,
            source_files=dict(self.source_files),
            n_rows_in_source=self.n_rows_in_source,
            notes={**self.notes, "subsampled_to": len(index), "subsample_seed": seed},
        )
