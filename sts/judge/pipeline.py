"""Orchestration for the LLM-as-judge experiment.

Mirrors :func:`sts.pipeline.run_experiment` — same datasets, same manifest,
same metrics and report layers, same resume semantics — with the encoder
replaced by :class:`LLMJudge` and the embedding cache reused as a judgment
cache (one row per ordered pair: the raw digit probabilities plus bookkeeping).

Per (judge, dataset) the run records, beyond the correlations:

* ``prompt_hash`` — spec hash combined with the chat template, so a later
  prompt edit is visible in every prediction row and in the manifest;
* ``argmax`` correlations next to the expectation-based ones, which shows what
  the probability weighting buys over a plain single-digit answer;
* ``valid_mass`` statistics — how much probability the model put on the digits
  it was allowed to answer with. Rows below ``min_valid_mass`` are NaN, counted
  as degenerate, and excluded from the correlation;
* an order-symmetry diagnostic on a sample: the same pairs scored with the two
  sentences swapped, reported as mean |Δ| and the Spearman between orders.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..cache import EmbeddingCache
from ..config import ExperimentConfig, JudgeSpec, get_logger, resolve_device, resolve_dtype
from ..data import STSDataset, load_dataset
from ..manifest import Manifest
from ..metrics import compute_correlations
from ..pipeline import RunPaths, metric_rows, set_global_seed
from .prompts import PromptSpec, family_of, get_prompt
from .scorer import LLMJudge, ScoreBatch

_log = get_logger()

PREDICTION_COLUMN = "expected_score"
#: Cache key = sentence_a + SEP + sentence_b. Order-sensitive on purpose: the
#: swapped order is a different prompt and gets its own row.
_PAIR_SEPARATOR = "\x1f"


def _pair_keys(sentence_a: Sequence[str], sentence_b: Sequence[str]) -> List[str]:
    return [f"{a}{_PAIR_SEPARATOR}{b}" for a, b in zip(sentence_a, sentence_b)]


# --------------------------------------------------------------------------- #
# Scoring through the cache
# --------------------------------------------------------------------------- #
def score_pairs_cached(
    judge: LLMJudge,
    spec: PromptSpec,
    cache: EmbeddingCache,
    sentence_a: Sequence[str],
    sentence_b: Sequence[str],
    flush_every: int,
) -> ScoreBatch:
    """Judge every (a, b) pair, consulting and filling the cache.

    Cached row layout: ``[p(digit) for each scale value, space_mass, prompt_tokens]``.
    """
    keys = _pair_keys(sentence_a, sentence_b)
    k = len(spec.scale)
    matrix, missing = cache.lookup(keys)

    if missing:
        _log.info("  cache: %d/%d pairs hit, judging %d", len(keys) - len(missing), len(keys), len(missing))
        computed: List[np.ndarray] = []
        for start in range(0, len(missing), flush_every):
            positions = missing[start : start + flush_every]
            prompts = [judge.render(spec, sentence_a[i], sentence_b[i]) for i in positions]
            batch = judge.score(prompts, spec.scale)
            rows = np.concatenate(
                [batch.raw_probs, batch.space_mass[:, None], batch.prompt_tokens[:, None].astype(np.float64)],
                axis=1,
            ).astype(np.float32)
            cache.store([keys[i] for i in positions], rows)
            cache.flush()
            computed.append(rows)
        new_rows = np.concatenate(computed, axis=0)
        if matrix is None:
            matrix = np.zeros((len(keys), k + 2), dtype=np.float32)
        elif matrix.shape[1] != k + 2:
            raise ValueError(
                f"cache rows are {matrix.shape[1]}-wide but this scale needs {k + 2}; "
                f"clear {cache.directory} and rerun"
            )
        for position, row in zip(missing, new_rows):
            matrix[position] = row
    else:
        _log.info("  cache: %d/%d pairs hit, nothing to judge", len(keys), len(keys))

    matrix = matrix.astype(np.float64)
    return ScoreBatch(
        raw_probs=matrix[:, :k],
        space_mass=matrix[:, k],
        prompt_tokens=matrix[:, k + 1].astype(np.int32),
    )


# --------------------------------------------------------------------------- #
# Predictions table
# --------------------------------------------------------------------------- #
def build_predictions(
    dataset: STSDataset,
    judge_spec: JudgeSpec,
    spec: PromptSpec,
    prompt_hash: str,
    forward: ScoreBatch,
    reverse: Optional[ScoreBatch],
) -> pd.DataFrame:
    frame = dataset.to_dataframe()
    scale = spec.scale
    min_mass = judge_spec.min_valid_mass

    expected_fwd = forward.expected(scale, min_mass)
    argmax_fwd = forward.argmax(scale, min_mass)
    norm_fwd = forward.normalised(min_mass)

    if reverse is not None:
        expected_rev = reverse.expected(scale, min_mass)
        expected = np.nanmean(np.stack([expected_fwd, expected_rev]), axis=0)
        expected[np.isnan(expected_fwd) & np.isnan(expected_rev)] = np.nan
    else:
        expected_rev = None
        expected = expected_fwd

    columns: Dict[str, object] = {
        "pair_id": frame["pair_id"],
        "dataset": frame["dataset"] + "-" + frame["split"],
        "split": frame["split"],
        "domain": frame["domain"],
        "model": judge_spec.key,
        "gold_score": frame["gold_score"],
        PREDICTION_COLUMN: expected,
        "argmax_score": argmax_fwd,
    }
    for index, value in enumerate(scale):
        columns[f"p_{value}"] = norm_fwd[:, index]
    columns.update({
        "valid_mass": forward.valid_mass,
        "space_mass": forward.space_mass,
        "prompt_tokens": forward.prompt_tokens,
        "pair_truncated": forward.prompt_tokens > judge_spec.max_prompt_tokens,
        "scale_min": min(scale),
        "scale_max": max(scale),
        "prompt_hash": prompt_hash,
        "sentence1_id": frame["sentence1_id"],
        "sentence2_id": frame["sentence2_id"],
        "image1_id": frame["image1_id"],
        "image2_id": frame["image2_id"],
    })
    if expected_rev is not None:
        columns["expected_score_fwd"] = expected_fwd
        columns["expected_score_rev"] = expected_rev
    return pd.DataFrame(columns)


# --------------------------------------------------------------------------- #
# Diagnostics recorded per (judge, dataset)
# --------------------------------------------------------------------------- #
def _finite_pair(a: np.ndarray, b: np.ndarray):
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask], int(mask.sum())


def answer_diagnostics(predictions: pd.DataFrame, scale: Sequence[int]) -> Dict[str, object]:
    """How the judge behaved: digit mass, ties, and expectation vs argmax."""
    gold = predictions["gold_score"].to_numpy(dtype=np.float64)
    expected = predictions[PREDICTION_COLUMN].to_numpy(dtype=np.float64)
    argmax = predictions["argmax_score"].to_numpy(dtype=np.float64)
    mass = predictions["valid_mass"].to_numpy(dtype=np.float64)

    e, g, n_e = _finite_pair(expected, gold)
    a, g2, n_a = _finite_pair(argmax, gold)
    ties_argmax = pd.Series(a).value_counts()
    ties_expected = pd.Series(np.round(e, 6)).value_counts()

    return {
        "n_degenerate": int(np.isnan(expected).sum()),
        "valid_mass_mean": float(np.mean(mass)),
        "valid_mass_median": float(np.median(mass)),
        "valid_mass_min": float(np.min(mass)),
        "space_mass_mean": float(predictions["space_mass"].mean()),
        "spearman_expected": compute_correlations(e, g).spearman if n_e > 1 else float("nan"),
        "spearman_argmax": compute_correlations(a, g2).spearman if n_a > 1 else float("nan"),
        "pearson_expected": compute_correlations(e, g).pearson if n_e > 1 else float("nan"),
        "pearson_argmax": compute_correlations(a, g2).pearson if n_a > 1 else float("nan"),
        "argmax_distinct_values": int(ties_argmax.size),
        "argmax_largest_tie_fraction": float(ties_argmax.iloc[0] / n_a) if n_a else float("nan"),
        "expected_distinct_values": int(ties_expected.size),
        "argmax_histogram": {str(int(v)): int(c) for v, c in ties_argmax.sort_index().items()},
        "scale": list(scale),
    }


def symmetry_check(
    judge: LLMJudge,
    spec: PromptSpec,
    cache: EmbeddingCache,
    dataset: STSDataset,
    forward_expected: np.ndarray,
    sample: int,
    seed: int,
    flush_every: int,
    min_valid_mass: float,
) -> Dict[str, object]:
    """Re-score a sample with the sentences swapped and measure the disagreement.

    A symmetric prompt does not make an LLM order-invariant; this quantifies
    how far from invariant the judge actually is on this dataset.
    """
    n = min(sample, len(dataset))
    if n < 2:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(dataset), size=n, replace=False))
    s1 = [dataset.pairs[i].sentence1 for i in index]
    s2 = [dataset.pairs[i].sentence2 for i in index]
    reverse = score_pairs_cached(judge, spec, cache, s2, s1, flush_every)
    e_rev = reverse.expected(spec.scale, min_valid_mass)
    e_fwd = forward_expected[index]
    a, b, n_ok = _finite_pair(e_fwd, e_rev)
    if n_ok < 2:
        return {"n": n, "n_finite": n_ok}
    delta = np.abs(a - b)
    return {
        "n": n,
        "n_finite": n_ok,
        "mean_abs_delta": float(delta.mean()),
        "median_abs_delta": float(np.median(delta)),
        "max_abs_delta": float(delta.max()),
        "spearman_between_orders": compute_correlations(a, b).spearman,
        "scale_width": float(max(spec.scale) - min(spec.scale)),
    }


def dump_prompt(
    output_dir: Path, spec: PromptSpec, judge: LLMJudge, prompt_hash: str, example: STSDataset
) -> Path:
    """Write the spec and one fully rendered prompt so a hash can be traced back."""
    directory = output_dir / "prompts"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.family}__{spec.version}__{prompt_hash[:12]}.md"
    if path.exists():
        return path
    pair = example.pairs[0]
    rendered = judge.render(spec, pair.sentence1, pair.sentence2)
    body = "\n".join([
        f"# Prompt {spec.family} {spec.version}",
        "",
        f"- prompt_hash (spec + chat template): `{prompt_hash}`",
        f"- spec_hash: `{spec.spec_hash}`",
        f"- chat_template_sha256: `{judge.template_hash}`",
        f"- judge: `{judge.model_id}`",
        f"- enable_thinking: {judge.enable_thinking}",
        f"- scale: {list(spec.scale)}",
        "",
        "## Source of the anchors",
        "",
        spec.source,
        "",
        "## Spec (JSON)",
        "",
        "```json",
        json.dumps(spec.to_dict(), indent=2, ensure_ascii=False),
        "```",
        "",
        f"## Rendered example (pair `{pair.pair_id}`), exactly as tokenised",
        "",
        "```",
        rendered,
        "```",
        "",
    ])
    path.write_text(body, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_judge_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Every enabled (judge, dataset) combination. Returns metric rows."""
    set_global_seed(config.runtime.seed)
    paths = RunPaths(config.output_dir)
    paths.ensure()

    manifest = Manifest(config=config, path=config.tagged("manifest", ".json"))
    manifest.payload["method"] = "llm-judge"
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.runtime.dtype, device)
    _log.info("Device=%s dtype=%s | %d judge(s) x %d dataset(s)",
              device, dtype, len(config.enabled_judges), len(config.datasets))

    datasets: Dict[str, STSDataset] = {}
    for name in config.datasets:
        dataset = load_dataset(name, config.data)
        if config.limit:
            dataset = dataset.subsample(config.limit, seed=config.runtime.seed)
        datasets[name] = dataset
        manifest.record_dataset(dataset)
        pair_path = paths.pair_file(name)
        if not pair_path.exists() or not config.resume:
            dataset.to_dataframe().to_csv(pair_path, index=False)

    all_metrics: List[Dict[str, object]] = []
    for judge_spec in config.enabled_judges:
        judge: Optional[LLMJudge] = None
        judge_dtype = resolve_dtype(judge_spec.dtype or config.runtime.dtype, device)
        caches: Dict[str, EmbeddingCache] = {}
        try:
            for name, dataset in datasets.items():
                spec = get_prompt(judge_spec.prompt_set, name)
                output_path = paths.prediction_file(judge_spec.key, name)

                if config.resume and output_path.exists():
                    _log.info("[%s | %s] resume: reusing %s", judge_spec.key, name, output_path)
                    predictions = pd.read_csv(output_path)
                else:
                    if judge is None:
                        judge = LLMJudge(
                            judge_spec, device, judge_dtype,
                            judge_spec.batch_size or config.runtime.batch_size,
                            config.runtime.progress,
                        )
                        manifest.record_model(judge_spec.key, judge_spec.model_id, {
                            **judge.describe(),
                            "prompt_set": judge_spec.prompt_set,
                            "symmetrize": judge_spec.symmetrize,
                            "min_valid_mass": judge_spec.min_valid_mass,
                        })
                    prompt_hash = judge.prompt_hash(spec)
                    family = family_of(name)
                    if family not in caches:
                        caches[family] = EmbeddingCache(
                            cache_dir=config.runtime.cache_dir,
                            model_key=f"{judge_spec.key}__{family}",
                            fingerprint={
                                **judge_spec.fingerprint(),
                                "resolved_dtype": judge_dtype,
                                "prompt_hash": prompt_hash,
                                "scale": list(spec.scale),
                            },
                            enabled=config.runtime.use_cache,
                        )
                    cache = caches[family]
                    prompt_file = dump_prompt(config.output_dir, spec, judge, prompt_hash, dataset)

                    _log.info("[%s | %s] %d pairs, prompt %s (%s), scale %s",
                              judge_spec.key, name, len(dataset), spec.version, prompt_hash[:12], list(spec.scale))
                    started = time.time()
                    forward = score_pairs_cached(
                        judge, spec, cache, dataset.sentence1, dataset.sentence2, config.runtime.flush_every
                    )
                    reverse = None
                    if judge_spec.symmetrize:
                        reverse = score_pairs_cached(
                            judge, spec, cache, dataset.sentence2, dataset.sentence1, config.runtime.flush_every
                        )
                    predictions = build_predictions(dataset, judge_spec, spec, prompt_hash, forward, reverse)

                    symmetry = symmetry_check(
                        judge, spec, cache, dataset,
                        forward.expected(spec.scale, judge_spec.min_valid_mass),
                        judge_spec.symmetry_check_sample, config.runtime.seed,
                        config.runtime.flush_every, judge_spec.min_valid_mass,
                    ) if judge_spec.symmetry_check_sample and not judge_spec.symmetrize else {"n": 0}
                    elapsed = time.time() - started

                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    predictions.to_csv(output_path, index=False)
                    diagnostics = answer_diagnostics(predictions, spec.scale)
                    if diagnostics["valid_mass_median"] < 0.5:
                        _log.warning("[%s | %s] median digit mass is only %.3f — the judge is often "
                                     "not answering with a digit; inspect %s",
                                     judge_spec.key, name, diagnostics["valid_mass_median"], prompt_file)
                    manifest.record_run({
                        "model": judge_spec.key,
                        "dataset": name,
                        "n_pairs": len(dataset),
                        "n_unique_texts": len(dataset.unique_texts()),
                        "runtime_seconds": round(elapsed, 2),
                        "pairs_per_second": round(len(dataset) / elapsed, 2) if elapsed else None,
                        "cache": cache.stats.to_dict(),
                        "n_truncated_pairs": int(predictions["pair_truncated"].sum()),
                        "predictions_file": str(output_path),
                        "prompt_hash": prompt_hash,
                        "spec_hash": spec.spec_hash,
                        "prompt_version": spec.version,
                        "prompt_file": str(prompt_file),
                        "symmetrized": judge_spec.symmetrize,
                        "answer": diagnostics,
                        "symmetry_check": symmetry,
                    })
                    _log.info("[%s | %s] done in %.1fs | ρ expected=%.4f argmax=%.4f | mass median=%.3f "
                              "| degenerate=%d | order Δ=%.3f -> %s",
                              judge_spec.key, name, elapsed,
                              diagnostics["spearman_expected"], diagnostics["spearman_argmax"],
                              diagnostics["valid_mass_median"], diagnostics["n_degenerate"],
                              symmetry.get("mean_abs_delta", float("nan")), output_path)

                valid = predictions[predictions[PREDICTION_COLUMN].notna()]
                n_degenerate = int(len(predictions) - len(valid))
                rows = metric_rows(valid, name, judge_spec.key, config, prediction_column=PREDICTION_COLUMN)
                for row in rows:
                    row["n_degenerate"] = n_degenerate
                all_metrics.extend(rows)
        finally:
            for cache in caches.values():
                cache.flush()
            if judge is not None:
                judge.close()

    manifest.finish()
    return pd.DataFrame(all_metrics)
