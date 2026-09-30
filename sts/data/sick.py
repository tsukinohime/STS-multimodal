"""SICK-Relatedness adapter.

``SICK.txt`` is tab-separated with a header. The columns used here are
``pair_ID``, ``sentence_A``, ``sentence_B``, ``relatedness_score`` (1–5),
``SemEval_set`` (TRAIN | TRIAL | TEST) and ``sentence_{A,B}_dataset``
(FLICKR | SEMEVAL — the corpus each sentence was drawn from), which becomes the
pair ``domain``.

The entailment labels are not part of the pair table: this project scores
relatedness only. :func:`load_sick_labels` exposes them, keyed by the same
``pair_id``, for the post-hoc breakdown in :mod:`sts.breakdown`.

``split="test"`` reproduces the standard SICK-R benchmark; ``split="all"``
covers every pair in the file. Both are reported, and note that they overlap —
``all`` contains ``test`` — so only one of them should feed the macro average.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import pandas as pd

from .. import archive
from ..config import get_logger
from .base import STSDataset, STSPair, clean_text

_SPLIT_MAP = {"train": "TRAIN", "trial": "TRIAL", "test": "TEST"}
#: Per-pair NLI annotations shipped with SICK. ``entailment_label`` is the
#: pair-level gold label; ``entailment_AB`` / ``entailment_BA`` are directional.
NLI_COLUMNS = ("entailment_label", "entailment_AB", "entailment_BA")
_log = get_logger()


def _pair_domain(dataset_a: str, dataset_b: str) -> str:
    """Source corpus of the pair, order-independent.

    Sorted so a FLICKR/SEMEVAL pair lands in one category regardless of which
    sentence came from which corpus.
    """
    a, b = str(dataset_a).strip(), str(dataset_b).strip()
    return a if a == b else "+".join(sorted((a, b)))


def _read_split(path: Path, split: str) -> Tuple[pd.DataFrame, int]:
    """The rows of ``split`` plus the row count of the whole file."""
    if not archive.exists(path):
        raise FileNotFoundError(f"SICK file not found: {path}")

    with archive.open_binary(path) as handle:
        frame = pd.read_csv(handle, sep="\t")
    missing_columns = {"pair_ID", "sentence_A", "sentence_B", "relatedness_score"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing columns: {sorted(missing_columns)}")

    n_rows_in_file = len(frame)
    if split != "all":
        if split not in _SPLIT_MAP:
            raise ValueError(f"unknown SICK split {split!r}; use all/train/trial/test")
        if "SemEval_set" not in frame.columns:
            raise ValueError(f"{path} has no 'SemEval_set' column; only split='all' is possible")
        frame = frame[frame["SemEval_set"] == _SPLIT_MAP[split]]
    return frame, n_rows_in_file


def load_sick(path: Union[str, Path], split: str = "test") -> STSDataset:
    path = Path(path)
    split = split.lower()
    frame, n_rows_in_file = _read_split(path, split)

    has_domain = {"sentence_A_dataset", "sentence_B_dataset"} <= set(frame.columns)
    name = f"sick-{split}"

    pairs = [
        STSPair(
            pair_id=f"{name}-{row.pair_ID}",
            dataset="sick",
            split=split,
            domain=_pair_domain(row.sentence_A_dataset, row.sentence_B_dataset) if has_domain else "",
            sentence1_id=f"sick:{row.pair_ID}:A",
            sentence2_id=f"sick:{row.pair_ID}:B",
            sentence1=clean_text(row.sentence_A),
            sentence2=clean_text(row.sentence_B),
            gold_score=float(row.relatedness_score),
            image1_id="",
            image2_id="",
        )
        for row in frame.itertuples(index=False)
    ]

    _log.info("Loaded %s: %d pairs", name, len(pairs))
    return STSDataset(
        name=name,
        split=split,
        pairs=pairs,
        score_range=(1.0, 5.0),
        source_files={archive.display(path): archive.sha256(path)},
        # Rows belonging to this split, i.e. what the pair count must equal.
        n_rows_in_source=len(frame),
        notes={
            "gold_column": "relatedness_score",
            "domain_column": "sentence_A_dataset/sentence_B_dataset" if has_domain else None,
            "semeval_set_filter": _SPLIT_MAP.get(split, "ALL"),
            "n_rows_in_file": n_rows_in_file,
        },
    )


def load_sick_labels(
    path: Union[str, Path], split: str = "all", column: str = "entailment_label"
) -> pd.Series:
    """One NLI annotation per pair, indexed by the ``pair_id`` :func:`load_sick` assigns."""
    if column not in NLI_COLUMNS:
        raise ValueError(f"unknown SICK label column {column!r}; use one of {NLI_COLUMNS}")
    path = Path(path)
    split = split.lower()
    frame, _ = _read_split(path, split)
    if column not in frame.columns:
        raise ValueError(f"{path} has no {column!r} column")
    labels = frame[column].astype(str).str.strip()
    labels.index = [f"sick-{split}-{pair_id}" for pair_id in frame["pair_ID"]]
    labels.name = column
    return labels
