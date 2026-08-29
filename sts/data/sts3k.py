"""STS3k adapter.

The distributed file in ``data/raw/sts3k/`` is ``STS3k_all.txt``: 2,800 lines of
``sentence1;sentence2;score`` with no header and no category column, scores in
[0, 1].

STS3k is published elsewhere with a per-pair category / condition column. The
loader therefore sniffs the file: if it parses as a delimited table carrying a
recognised category column, that column becomes the pair ``domain`` and the
report gains per-category rows automatically. With the plain 3-field file every
pair gets an empty domain and only the overall numbers are reported — the
project brief asks for category results *when the official data provides them*.

STS3k ships as a single undivided file, so the split is ``all``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

from .. import archive
from ..config import get_logger
from .base import STSDataset, STSPair, clean_text

_log = get_logger()

#: Column names that different STS3k releases use for the pair category.
_CATEGORY_COLUMNS = ("category", "condition", "type", "pair_type", "class", "group", "subset")
_SENTENCE1_COLUMNS = ("sentence1", "sentence_1", "sent1", "s1", "sentence_a", "text1")
_SENTENCE2_COLUMNS = ("sentence2", "sentence_2", "sent2", "s2", "sentence_b", "text2")
_SCORE_COLUMNS = ("score", "gold", "similarity", "sts", "mean", "rating", "gold_score")


def _find_column(columns: List[str], candidates) -> Optional[str]:
    lowered = {c.strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def _head_text(path: Path, n_bytes: int = 8192) -> str:
    with archive.open_binary(path) as handle:
        return handle.read(n_bytes).decode("utf-8", errors="replace")


def _sniff_delimiter(path: Path) -> str:
    try:
        return csv.Sniffer().sniff(_head_text(path), delimiters=";,\t").delimiter
    except csv.Error:
        return ";"


def _has_header(path: Path, delimiter: str) -> bool:
    """True when the first line is a header rather than a data row.

    A data row always ends in a numeric score, so a non-numeric last field
    identifies the header without relying on csv.Sniffer heuristics.
    """
    first = _head_text(path).split("\n", 1)[0].rstrip("\r")
    if not first:
        return False
    try:
        float(first.split(delimiter)[-1])
    except ValueError:
        return True
    return False


def load_sts3k(path: Union[str, Path], split: str = "all") -> STSDataset:
    path = Path(path)
    if not archive.exists(path):
        raise FileNotFoundError(f"STS3k file not found: {path}")

    delimiter = _sniff_delimiter(path)
    header_row = 0 if _has_header(path, delimiter) else None
    with archive.open_binary(path) as handle:
        frame = pd.read_csv(handle, sep=delimiter, header=header_row, engine="python")

    if header_row is None:
        if frame.shape[1] < 3:
            raise ValueError(
                f"{path}: expected at least 3 '{delimiter}'-separated fields, got {frame.shape[1]}"
            )
        col_s1, col_s2, col_score = frame.columns[0], frame.columns[1], frame.columns[2]
        col_category = frame.columns[3] if frame.shape[1] > 3 else None
    else:
        columns = [str(c) for c in frame.columns]
        col_s1 = _find_column(columns, _SENTENCE1_COLUMNS) or columns[0]
        col_s2 = _find_column(columns, _SENTENCE2_COLUMNS) or columns[1]
        col_score = _find_column(columns, _SCORE_COLUMNS) or columns[2]
        col_category = _find_column(columns, _CATEGORY_COLUMNS)

    if col_category is not None:
        _log.info("STS3k: using column %r as the pair category", col_category)
    else:
        _log.info("STS3k: no category column in %s; reporting overall results only", path.name)

    name = f"sts3k-{split}" if split != "all" else "sts3k"
    pairs = [
        STSPair(
            pair_id=f"{name}-{row_number:05d}",
            dataset="sts3k",
            split=split,
            domain=clean_text(row[col_category]) if col_category is not None else "",
            sentence1_id=f"sts3k:{row_number}:1",
            sentence2_id=f"sts3k:{row_number}:2",
            sentence1=clean_text(row[col_s1]),
            sentence2=clean_text(row[col_s2]),
            gold_score=float(row[col_score]),
            image1_id="",
            image2_id="",
        )
        for row_number, (_, row) in enumerate(frame.iterrows(), start=1)
    ]

    scores = [p.gold_score for p in pairs]
    _log.info("Loaded %s: %d pairs (gold %.3f–%.3f)", name, len(pairs), min(scores), max(scores))
    return STSDataset(
        name=name,
        split=split,
        pairs=pairs,
        score_range=(0.0, 1.0),
        source_files={archive.display(path): archive.sha256(path)},
        n_rows_in_source=len(frame),
        notes={
            "gold_column": str(col_score),
            "domain_column": str(col_category) if col_category is not None else None,
            "delimiter": delimiter,
            "had_header": header_row == 0,
        },
    )
