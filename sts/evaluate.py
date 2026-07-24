"""Evaluation orchestration: load -> encode -> cosine -> correlate.

The heavy step is encoding, so texts are de-duplicated before encoding (CxC in
particular reuses the same captions across many pairs), then re-indexed back to
the pair layout.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from .config import Config, get_logger
from .data import STSData, load_cxc_sts, load_coco_karpathy_sentid_map, load_sick
from .encoders.base import TextEncoder
from .metrics import Correlations, compute_correlations
from .similarity import paired_cosine_similarity

_log = get_logger()


# --------------------------------------------------------------------------- #
# Dataset registry
# --------------------------------------------------------------------------- #
def available_datasets() -> List[str]:
    return ["sick", "cxc-val", "cxc-test"]


def build_dataset(name: str, config: Config) -> STSData:
    """Resolve a dataset name (optionally with a ``sick:split`` suffix) to STSData."""
    key = name.lower()

    if key.startswith("sick"):
        split = key.split(":", 1)[1] if ":" in key else config.sick_split
        return load_sick(config.sick_path, split=split)

    if key in ("cxc-val", "cxc_val", "cxc-test", "cxc_test"):
        split = "val" if "val" in key else "test"
        sentid_map = _load_sentid_map(str(config.coco_karpathy_json))
        csv_path = config.cxc_data_dir / f"sts_{split}.csv"
        return load_cxc_sts(csv_path, sentid_map)

    raise ValueError(f"unknown dataset {name!r}; available: {available_datasets()}")


@lru_cache(maxsize=2)
def _load_sentid_map(json_path: str):
    # Cached so val + test share one parse of the large Karpathy json.
    return load_coco_karpathy_sentid_map(json_path)


# --------------------------------------------------------------------------- #
# Encoding + scoring
# --------------------------------------------------------------------------- #
def _encode_paired(encoder: TextEncoder, text_a: List[str], text_b: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Encode unique texts once, then re-index to the per-pair layout."""
    unique = list(dict.fromkeys(list(text_a) + list(text_b)))
    index = {t: i for i, t in enumerate(unique)}
    _log.info("Encoding %d unique texts (from %d pairs)", len(unique), len(text_a))
    emb = encoder.encode(unique)
    emb_a = emb[[index[t] for t in text_a]]
    emb_b = emb[[index[t] for t in text_b]]
    return emb_a, emb_b


def evaluate_dataset(data: STSData, encoder: TextEncoder, with_kendall: bool = False) -> Dict:
    emb_a, emb_b = _encode_paired(encoder, data.text_a, data.text_b)
    preds = paired_cosine_similarity(emb_a, emb_b)
    corr: Correlations = compute_correlations(preds, data.gold, with_kendall=with_kendall)
    result = {"dataset": data.name, **corr.to_dict()}
    _log.info(
        "%s  |  n=%d  spearman=%.4f  pearson=%.4f",
        data.name, corr.n, corr.spearman, corr.pearson,
    )
    return result


def run_evaluation(config: Config, dataset_names: List[str], with_kendall: bool = False) -> List[Dict]:
    """Full run over one encoder and several datasets. Returns a list of result rows."""
    encoder = _build_encoder_cached(config)
    results: List[Dict] = []
    for name in dataset_names:
        data = build_dataset(name, config)
        if config.limit:
            data = data.subsample(config.limit, seed=config.seed)
        results.append(evaluate_dataset(data, encoder, with_kendall=with_kendall))
    if results:
        mean_spearman = float(np.mean([r["spearman"] for r in results]))
        mean_pearson = float(np.mean([r["pearson"] for r in results]))
        results.append({"dataset": "MEAN", "n": None, "spearman": mean_spearman, "pearson": mean_pearson})
    return results


def _build_encoder_cached(config: Config) -> TextEncoder:
    from .encoders.base import build_encoder

    return build_encoder(config)
