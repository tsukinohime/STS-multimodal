"""Validation checks and the cross-model alignment probe.

Two groups of things live here:

**Correctness checks** — the invariants a zero-shot cosine STS pipeline must
satisfy before any number it prints is worth reading. They are cheap and run on
the smoke config.

**Cross-model alignment** — a measurement, not a check. ``v5-omni-small``'s
model card states its embeddings share a vector space with ``v5-text-small``,
which would allow encoding text with the cheaper text tower and images with the
omni tower. That claim is quantified here rather than assumed: how close the
two models place the *same* sentence, and how much STS accuracy is lost when
the two sides of a pair are encoded by different models.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig, get_logger
from .data import STSDataset, load_dataset
from .data.cxc import _parse_sentid, load_coco_karpathy_index
from .metrics import compute_correlations
from .models.base import TextEncoder, build_encoder
from .similarity import paired_cosine_similarity

_log = get_logger()

#: Cosine of a vector with itself must be this close to 1.0.
IDENTITY_TOLERANCE = 1e-4
#: cos(a,b) vs cos(b,a) must agree to this; the operation is symmetric by
#: construction, so any drift means non-determinism in the encoder.
SYMMETRY_TOLERANCE = 1e-6


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    data: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail, **self.data}


# --------------------------------------------------------------------------- #
# Data-level checks (no model needed)
# --------------------------------------------------------------------------- #
def check_pair_counts(datasets: Dict[str, STSDataset], limited: bool) -> CheckResult:
    """Pair count matches the number of rows in the source file."""
    rows = []
    ok = True
    for name, dataset in datasets.items():
        expected = dataset.n_rows_in_source
        actual = len(dataset)
        subsampled = dataset.notes.get("subsampled_to")
        matches = actual == expected if subsampled is None else actual == subsampled
        ok = ok and matches
        rows.append({
            "dataset": name,
            "pairs_loaded": actual,
            "rows_in_source": expected,
            "subsampled_to": subsampled,
            "match": matches,
        })
    note = " (counts compared against the subsample size; run without --limit for the full check)" if limited else ""
    return CheckResult(
        name="pair_count_matches_source",
        passed=ok,
        detail=("every dataset has as many pairs as its source file" + note) if ok
        else "pair count differs from the source file",
        data={"per_dataset": rows},
    )


def check_no_nan_gold(datasets: Dict[str, STSDataset]) -> CheckResult:
    offenders = {
        name: int(np.isnan(dataset.gold).sum())
        for name, dataset in datasets.items()
        if np.isnan(dataset.gold).any()
    }
    return CheckResult(
        name="gold_scores_have_no_nan",
        passed=not offenders,
        detail="no NaN gold score" if not offenders else f"NaN gold scores: {offenders}",
        data={"offenders": offenders},
    )


def check_cxc_caption_mapping(config: ExperimentConfig) -> CheckResult:
    """Every CxC caption id resolves to COCO caption text and an image id."""
    files = [
        ("cxc-val", config.data.cxc_dir / "sts_val.csv"),
        ("cxc-test", config.data.cxc_dir / "sts_test.csv"),
    ]
    files = [(name, path) for name, path in files if path.exists()]
    if not files:
        return CheckResult("cxc_caption_ids_resolve", True, "no CxC files present; skipped")

    text_by_sentid, image_by_sentid = load_coco_karpathy_index(str(config.data.coco_karpathy_json))
    report = []
    ok = True
    for name, path in files:
        frame = pd.read_csv(path)
        ids = set(frame["caption1"]) | set(frame["caption2"])
        sentids = {_parse_sentid(i) for i in ids}
        unmapped_text = [s for s in sentids if s not in text_by_sentid]
        unmapped_image = [s for s in sentids if s not in image_by_sentid]
        ok = ok and not unmapped_text and not unmapped_image
        report.append({
            "dataset": name,
            "rows": len(frame),
            "unique_caption_ids": len(ids),
            "unmapped_to_text": len(unmapped_text),
            "unmapped_to_image": len(unmapped_image),
        })
    return CheckResult(
        name="cxc_caption_ids_resolve",
        passed=ok,
        detail="all CxC caption ids map to COCO text and an image id" if ok
        else "some CxC caption ids did not resolve",
        data={"per_file": report},
    )


def check_predictions(output_dir: Path) -> CheckResult:
    """No NaN in any written prediction, and cosine stays inside [-1, 1]."""
    files = sorted((output_dir / "predictions").rglob("*.csv"))
    if not files:
        return CheckResult("predictions_have_no_nan", False, "no prediction files found")

    problems = []
    total = 0
    for path in files:
        frame = pd.read_csv(path)
        total += len(frame)
        n_nan_pred = int(frame["cosine_prediction"].isna().sum())
        n_nan_gold = int(frame["gold_score"].isna().sum())
        out_of_range = int((frame["cosine_prediction"].abs() > 1.0 + 1e-6).sum())
        if n_nan_pred or n_nan_gold or out_of_range:
            problems.append({
                "file": str(path.relative_to(output_dir)),
                "nan_predictions": n_nan_pred,
                "nan_gold": n_nan_gold,
                "cosine_out_of_range": out_of_range,
            })
    return CheckResult(
        name="predictions_have_no_nan",
        passed=not problems,
        detail=f"{total} pair predictions across {len(files)} file(s), all finite and in [-1, 1]"
        if not problems else f"problems in {len(problems)} file(s)",
        data={"n_files": len(files), "n_pairs": total, "problems": problems},
    )


# --------------------------------------------------------------------------- #
# Encoder-level checks
# --------------------------------------------------------------------------- #
def check_identical_sentences(encoder: TextEncoder, texts: Sequence[str]) -> CheckResult:
    """cos(x, x) must be 1 — catches broken pooling and non-deterministic encoding.

    The two copies are encoded in a single batch at different positions, so a
    padding or position-dependent bug would show up as a cosine below 1.
    """
    texts = list(texts)
    output = encoder.encode(texts + texts)
    n = len(texts)
    cosines = paired_cosine_similarity(output.embeddings[:n], output.embeddings[n:])
    worst = float(np.min(cosines))
    passed = bool(np.all(np.abs(cosines - 1.0) <= IDENTITY_TOLERANCE))
    return CheckResult(
        name=f"identical_sentences_cosine_is_1[{encoder.key}]",
        passed=passed,
        detail=f"min cos(x,x) = {worst:.8f} over {n} sentences (tolerance {IDENTITY_TOLERANCE})",
        data={"model": encoder.key, "n": n, "min_cosine": worst, "mean_cosine": float(np.mean(cosines))},
    )


def check_order_symmetry(
    encoder: TextEncoder, sentence1: Sequence[str], sentence2: Sequence[str]
) -> CheckResult:
    """cos(a, b) == cos(b, a) — swapping the sentence order must change nothing."""
    forward_a = encoder.encode(list(sentence1)).embeddings
    forward_b = encoder.encode(list(sentence2)).embeddings
    forward = paired_cosine_similarity(forward_a, forward_b)
    # Re-encode in the swapped batch layout rather than just transposing the
    # matrices, so batching effects would also be caught.
    swapped_a = encoder.encode(list(sentence2)).embeddings
    swapped_b = encoder.encode(list(sentence1)).embeddings
    swapped = paired_cosine_similarity(swapped_a, swapped_b)

    max_delta = float(np.max(np.abs(forward - swapped)))
    return CheckResult(
        name=f"similarity_is_order_invariant[{encoder.key}]",
        passed=max_delta <= SYMMETRY_TOLERANCE,
        detail=f"max |cos(a,b) - cos(b,a)| = {max_delta:.3e} over {len(forward)} pairs "
               f"(tolerance {SYMMETRY_TOLERANCE})",
        data={"model": encoder.key, "n": len(forward), "max_abs_delta": max_delta},
    )


def check_embedding_norms(encoder: TextEncoder, texts: Sequence[str]) -> CheckResult:
    output = encoder.encode(list(texts))
    norms = np.linalg.norm(output.embeddings, axis=1)
    deviation = float(np.max(np.abs(norms - 1.0)))
    return CheckResult(
        name=f"embeddings_are_l2_normalized[{encoder.key}]",
        passed=deviation <= 1e-4,
        detail=f"max |‖v‖ - 1| = {deviation:.3e}; dim = {output.embeddings.shape[1]}",
        data={"model": encoder.key, "dim": int(output.embeddings.shape[1]), "max_norm_deviation": deviation},
    )


# --------------------------------------------------------------------------- #
# Cross-model alignment
# --------------------------------------------------------------------------- #
def cross_model_alignment(
    config: ExperimentConfig,
    model_a_key: str,
    model_b_key: str,
    dataset_name: str,
    sample: int = 512,
) -> Dict[str, object]:
    """Quantify whether two encoders really share one embedding space.

    Reports three things on the same pairs:

    * ``same_text_cosine`` — cosine between model A's and model B's embedding of
      the *identical* sentence. 1.0 means the spaces coincide.
    * ``spearman_a`` / ``spearman_b`` — normal same-model STS accuracy.
    * ``spearman_cross`` — STS accuracy when sentence 1 is encoded by A and
      sentence 2 by B. This is the configuration a mixed text/image setup
      implies, and the gap against the same-model numbers is the cost of mixing.
    """
    spec_a = config.model_by_key(model_a_key)
    spec_b = config.model_by_key(model_b_key)

    dataset = load_dataset(dataset_name, config.data)
    if sample and sample < len(dataset):
        dataset = dataset.subsample(sample, seed=config.runtime.seed)

    sentence1, sentence2 = dataset.sentence1, dataset.sentence2
    gold = dataset.gold

    embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    dims: Dict[str, int] = {}
    for spec in (spec_a, spec_b):
        encoder = build_encoder(spec, config.runtime)
        try:
            embeddings[spec.key] = {
                "s1": encoder.encode(sentence1).embeddings,
                "s2": encoder.encode(sentence2).embeddings,
            }
            dims[spec.key] = embeddings[spec.key]["s1"].shape[1]
        finally:
            encoder.close()

    if dims[spec_a.key] != dims[spec_b.key]:
        raise ValueError(
            f"{spec_a.key} is {dims[spec_a.key]}-dim but {spec_b.key} is {dims[spec_b.key]}-dim; "
            "they cannot share a space"
        )

    same_text = np.concatenate([
        paired_cosine_similarity(embeddings[spec_a.key]["s1"], embeddings[spec_b.key]["s1"]),
        paired_cosine_similarity(embeddings[spec_a.key]["s2"], embeddings[spec_b.key]["s2"]),
    ])

    def spearman(pred) -> float:
        return compute_correlations(pred, gold).spearman

    pred_a = paired_cosine_similarity(embeddings[spec_a.key]["s1"], embeddings[spec_a.key]["s2"])
    pred_b = paired_cosine_similarity(embeddings[spec_b.key]["s1"], embeddings[spec_b.key]["s2"])
    pred_cross = paired_cosine_similarity(embeddings[spec_a.key]["s1"], embeddings[spec_b.key]["s2"])

    spearman_a, spearman_b, spearman_cross = spearman(pred_a), spearman(pred_b), spearman(pred_cross)
    return {
        "model_a": spec_a.key,
        "model_b": spec_b.key,
        "model_a_id": spec_a.model_id,
        "model_b_id": spec_b.model_id,
        "dataset": dataset_name,
        "n_pairs": len(dataset),
        "dim": dims[spec_a.key],
        "same_text_cosine": {
            "mean": float(np.mean(same_text)),
            "median": float(np.median(same_text)),
            "min": float(np.min(same_text)),
            "p05": float(np.percentile(same_text, 5)),
        },
        "spearman_a": spearman_a,
        "spearman_b": spearman_b,
        "spearman_cross_encoded": spearman_cross,
        "cross_encoding_penalty_vs_best": float(max(spearman_a, spearman_b) - spearman_cross),
        "prediction_agreement_spearman": compute_correlations(pred_a, pred_b).spearman,
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_validation(
    config: ExperimentConfig,
    encoder_checks: bool = True,
    sample: int = 16,
) -> List[CheckResult]:
    """Run every check and return the results in report order."""
    results: List[CheckResult] = []

    datasets = {name: load_dataset(name, config.data) for name in config.datasets}
    results.append(check_cxc_caption_mapping(config))
    results.append(check_pair_counts(
        {n: (d.subsample(config.limit, config.runtime.seed) if config.limit else d)
         for n, d in datasets.items()},
        limited=bool(config.limit),
    ))
    results.append(check_no_nan_gold(datasets))
    results.append(check_predictions(config.output_dir))

    if encoder_checks:
        probe_texts: List[str] = []
        probe_s1: List[str] = []
        probe_s2: List[str] = []
        per_dataset = max(1, sample // max(1, len(datasets)))
        for dataset in datasets.values():
            subset = dataset.subsample(per_dataset, seed=config.runtime.seed)
            probe_s1.extend(subset.sentence1)
            probe_s2.extend(subset.sentence2)
        probe_texts = list(dict.fromkeys(probe_s1 + probe_s2))[:sample]

        for spec in config.enabled_models:
            encoder = build_encoder(spec, config.runtime)
            try:
                results.append(check_embedding_norms(encoder, probe_texts))
                results.append(check_identical_sentences(encoder, probe_texts))
                results.append(check_order_symmetry(encoder, probe_s1, probe_s2))
            finally:
                encoder.close()

    return results


def write_validation_report(results: List[CheckResult], path: Path, extra: Optional[dict] = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_passed": all(r.passed for r in results),
        "n_checks": len(results),
        "n_failed": sum(1 for r in results if not r.passed),
        "checks": [r.to_dict() for r in results],
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _log.info("Validation report -> %s", path)
    return path


def print_validation(results: List[CheckResult]) -> bool:
    print("\nValidation")
    print("-" * 78)
    for result in results:
        mark = "PASS" if result.passed else "FAIL"
        print(f"  [{mark}] {result.name}")
        print(f"         {result.detail}")
    passed = all(r.passed for r in results)
    print("-" * 78)
    print(f"  {sum(1 for r in results if r.passed)}/{len(results)} checks passed\n")
    return passed
