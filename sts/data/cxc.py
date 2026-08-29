"""Crisscrossed Captions (CxC) — caption-to-caption STS only.

``sts_val.csv`` / ``sts_test.csv`` hold the caption–caption ratings and have
columns ``caption1, caption2, agg_score, sampling_method``. (The ``sis_*`` and
``sits_*`` files in the same directory are image–image and image–text
similarity; they are deliberately not touched at this stage.)

Caption ids look like ``COCO_val2014:sentid:190268``. The trailing integer is
the Karpathy-split ``sentid``, which ``dataset_coco.json`` maps to both the raw
caption text and the ``cocoid`` of the image it describes. That image id is
carried into the pair table so the later generated-image experiments can be
compared against the *real* COCO image for the same caption.

``sampling_method`` is used as the pair ``domain``:

* ``c2c_cocaption`` — the two captions describe the same COCO image.
* ``c2c_isim``      — the two captions come from different, visually similar images.

That split matters for the visual milestone, so it is reported separately.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from .. import archive
from ..config import get_logger
from .base import STSDataset, STSPair, clean_text

_SENTID_RE = re.compile(r"sentid:(\d+)")
_log = get_logger()


class CaptionMappingError(RuntimeError):
    """Raised when a CxC caption id cannot be resolved to COCO caption text."""


@lru_cache(maxsize=4)
def load_coco_karpathy_index(json_path: str) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Return ``({sentid: caption_text}, {sentid: cocoid})`` from the Karpathy json.

    Cached because the val and test splits share one ~180 MB parse. The path may
    resolve to a real file or to a member of a ``.tar`` beside it — see
    :mod:`sts.archive`.
    """
    with archive.open_binary(json_path) as handle:
        data = json.load(handle)

    text_by_sentid: Dict[int, str] = {}
    image_by_sentid: Dict[int, int] = {}
    for image in data["images"]:
        cocoid = int(image["cocoid"])
        for sentence in image["sentences"]:
            sentid = int(sentence["sentid"])
            text_by_sentid[sentid] = clean_text(sentence["raw"])
            image_by_sentid[sentid] = cocoid

    _log.info("Karpathy COCO index: %d captions over %d images",
              len(text_by_sentid), len(data["images"]))
    return text_by_sentid, image_by_sentid


def _parse_sentid(caption_id: str) -> int:
    match = _SENTID_RE.search(str(caption_id))
    if not match:
        raise CaptionMappingError(f"cannot parse sentid from caption id {caption_id!r}")
    return int(match.group(1))


def load_cxc_sts(
    csv_path: Union[str, Path],
    coco_karpathy_json: Union[str, Path],
    split: Optional[str] = None,
) -> STSDataset:
    csv_path = Path(csv_path)
    coco_karpathy_json = Path(coco_karpathy_json)
    if not archive.exists(csv_path):
        raise FileNotFoundError(f"CxC STS file not found: {csv_path}")
    if not archive.exists(coco_karpathy_json):
        raise FileNotFoundError(
            f"Karpathy COCO json not found: {coco_karpathy_json} "
            "(needed to resolve CxC caption ids to text)"
        )

    text_by_sentid, image_by_sentid = load_coco_karpathy_index(str(coco_karpathy_json))

    with archive.open_binary(csv_path) as handle:
        frame = pd.read_csv(handle)
    missing_columns = {"caption1", "caption2", "agg_score"} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{csv_path} is missing columns: {sorted(missing_columns)}")

    split = split or csv_path.stem.replace("sts_", "")  # "val" / "test"
    name = f"cxc-{split}"

    # Every caption id must resolve. Silently dropping one would quietly change
    # the benchmark, so an unmapped id is a hard error instead.
    unresolved: List[Tuple[str, str]] = []
    pairs: List[STSPair] = []
    for row_number, row in enumerate(frame.itertuples(index=False), start=1):
        sentid_a = _parse_sentid(row.caption1)
        sentid_b = _parse_sentid(row.caption2)
        if sentid_a not in text_by_sentid or sentid_b not in text_by_sentid:
            unresolved.append((str(row.caption1), str(row.caption2)))
            continue
        pairs.append(
            STSPair(
                pair_id=f"{name}-{row_number:06d}",
                dataset="cxc",
                split=split,
                domain=str(getattr(row, "sampling_method", "") or ""),
                sentence1_id=str(row.caption1),
                sentence2_id=str(row.caption2),
                sentence1=text_by_sentid[sentid_a],
                sentence2=text_by_sentid[sentid_b],
                gold_score=float(row.agg_score),
                image1_id=str(image_by_sentid[sentid_a]),
                image2_id=str(image_by_sentid[sentid_b]),
            )
        )

    if unresolved:
        raise CaptionMappingError(
            f"{csv_path.name}: {len(unresolved)} caption ids did not resolve against "
            f"{coco_karpathy_json.name} (first: {unresolved[0]}). "
            "The Karpathy split json is probably the wrong version."
        )

    _log.info("Loaded %s: %d pairs", name, len(pairs))
    return STSDataset(
        name=name,
        split=split,
        pairs=pairs,
        score_range=(0.0, 5.0),
        # Keys record where the bytes actually came from (a path, or
        # "<archive>::<member>"); hashes are of the content either way, so a
        # manifest from archived data is comparable with one from extracted data.
        source_files={
            archive.display(csv_path): archive.sha256(csv_path),
            archive.display(coco_karpathy_json): archive.sha256(coco_karpathy_json),
        },
        n_rows_in_source=len(frame),
        notes={
            "gold_column": "agg_score",
            "domain_column": "sampling_method",
            "caption_ids_resolved": len(pairs),
            "task": "caption-caption STS (sis_*/sits_* files intentionally unused)",
        },
    )
