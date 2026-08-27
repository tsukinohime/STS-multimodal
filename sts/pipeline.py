"""Run orchestration: load -> encode (cached) -> cosine -> correlate -> write.

One (model, dataset) combination is the unit of work and the unit of resume:
its pair-level predictions land in their own CSV, and a rerun with
``resume: true`` skips any combination whose CSV already exists. Below that,
the embedding cache flushes every ``flush_every`` unique texts, so even a
combination interrupted mid-encode restarts near where it stopped.

Models are loaded one at a time and released before the next, so a run over
several encoders never holds two sets of weights on the GPU at once.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .cache import EmbeddingCache
from .config import ExperimentConfig, ModelSpec, get_logger, resolve_device, resolve_dtype
from .data import STSDataset, load_dataset
from .manifest import Manifest
from .metrics import compute_correlations
from .models.base import TextEncoder, build_encoder
from .similarity import paired_cosine_similarity

_log = get_logger()

PREDICTION_COLUMNS = (
    "pair_id", "dataset", "split", "domain", "model",
    "gold_score", "cosine_prediction",
    "sentence1_id", "sentence2_id", "image1_id", "image2_id",
    "sentence1_tokens", "sentence2_tokens",
    "sentence1_truncated", "sentence2_truncated", "pair_truncated",
)


@dataclass
class RunPaths:
    root: Path

    @property
    def predictions(self) -> Path:
        return self.root / "predictions"

    @property
    def pairs(self) -> Path:
        return self.root / "pairs"

    def prediction_file(self, model_key: str, dataset_name: str) -> Path:
        return self.predictions / model_key / f"{dataset_name}.csv"

    def pair_file(self, dataset_name: str) -> Path:
        return self.pairs / f"{dataset_name}.csv"

    def ensure(self) -> None:
        for directory in (self.root, self.predictions, self.pairs):
            directory.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int) -> None:
    """Inference here is deterministic, but pin the seeds anyway.

    They do govern dataset subsampling and the metric bootstrap, both of which
    must reproduce exactly.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def encode_texts_cached(
    encoder: TextEncoder,
    texts: Sequence[str],
    cache: EmbeddingCache,
    flush_every: int,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Embed ``texts`` (already deduplicated), consulting and filling the cache.

    Returns the embedding matrix aligned with ``texts`` plus token bookkeeping.
    """
    texts = list(texts)
    matrix, missing = cache.lookup(texts)
    token_counts = encoder.token_lengths(texts)

    if not missing:
        _log.info("  cache: %d/%d texts hit, nothing to encode", len(texts), len(texts))
        return matrix, {"token_counts": token_counts, "truncated": token_counts > encoder.max_length}

    _log.info("  cache: %d/%d texts hit, encoding %d", len(texts) - len(missing), len(texts), len(missing))
    todo = [texts[i] for i in missing]

    computed: List[np.ndarray] = []
    for start in range(0, len(todo), flush_every):
        block = todo[start : start + flush_every]
        output = encoder.encode(block)
        computed.append(output.embeddings)
        cache.store(block, output.embeddings)
        cache.flush()
    new_vectors = np.concatenate(computed, axis=0)

    if matrix is None:
        matrix = np.zeros((len(texts), new_vectors.shape[1]), dtype=np.float32)
    elif matrix.shape[1] != new_vectors.shape[1]:
        raise ValueError(
            f"cached embeddings are {matrix.shape[1]}-dim but the encoder produced "
            f"{new_vectors.shape[1]}-dim vectors — clear {cache.directory} and rerun"
        )
    for position, vector in zip(missing, new_vectors):
        matrix[position] = vector

    return matrix, {"token_counts": token_counts, "truncated": token_counts > encoder.max_length}


def evaluate_model_on_dataset(
    encoder: TextEncoder,
    model_spec: ModelSpec,
    dataset: STSDataset,
    cache: EmbeddingCache,
    flush_every: int,
) -> pd.DataFrame:
    """Pair-level predictions for one (model, dataset)."""
    unique_texts = dataset.unique_texts()
    row_of = {text: i for i, text in enumerate(unique_texts)}

    embeddings, tokens = encode_texts_cached(encoder, unique_texts, cache, flush_every)

    index_a = [row_of[t] for t in dataset.sentence1]
    index_b = [row_of[t] for t in dataset.sentence2]
    predictions = paired_cosine_similarity(embeddings[index_a], embeddings[index_b])

    counts = tokens["token_counts"]
    truncated = tokens["truncated"]
    frame = dataset.to_dataframe()
    return pd.DataFrame(
        {
            "pair_id": frame["pair_id"],
            "dataset": frame["dataset"] + "-" + frame["split"],
            "split": frame["split"],
            "domain": frame["domain"],
            "model": model_spec.key,
            "gold_score": frame["gold_score"],
            "cosine_prediction": predictions,
            "sentence1_id": frame["sentence1_id"],
            "sentence2_id": frame["sentence2_id"],
            "image1_id": frame["image1_id"],
            "image2_id": frame["image2_id"],
            "sentence1_tokens": counts[index_a],
            "sentence2_tokens": counts[index_b],
            "sentence1_truncated": truncated[index_a],
            "sentence2_truncated": truncated[index_b],
            "pair_truncated": truncated[index_a] | truncated[index_b],
        },
        columns=list(PREDICTION_COLUMNS),
    )


def metric_rows(
    predictions: pd.DataFrame,
    dataset_name: str,
    model_key: str,
    config: ExperimentConfig,
) -> List[Dict[str, object]]:
    """Overall + per-domain correlation rows for one (model, dataset)."""
    settings = config.metrics
    rows: List[Dict[str, object]] = []

    def add(domain: str, frame: pd.DataFrame) -> None:
        correlations = compute_correlations(
            frame["cosine_prediction"].to_numpy(),
            frame["gold_score"].to_numpy(),
            bootstrap=settings.bootstrap,
            seed=settings.bootstrap_seed,
            confidence=settings.confidence,
        )
        rows.append({
            "model": model_key,
            "dataset": dataset_name,
            "split": frame["split"].iloc[0] if len(frame) else "",
            "domain": domain,
            **correlations.to_dict(),
            "n_truncated_pairs": int(frame["pair_truncated"].sum()),
        })

    add("ALL", predictions)
    if settings.report_domains:
        for domain, group in predictions.groupby("domain", sort=True):
            if domain:
                add(str(domain), group)
    return rows


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """Execute every enabled (model, dataset) combination. Returns metric rows."""
    set_global_seed(config.runtime.seed)
    paths = RunPaths(config.output_dir)
    paths.ensure()

    manifest = Manifest(config=config, path=config.tagged("manifest", ".json"))
    device = resolve_device(config.runtime.device)
    dtype = resolve_dtype(config.runtime.dtype, device)
    _log.info("Device=%s dtype=%s | %d model(s) x %d dataset(s)",
              device, dtype, len(config.enabled_models), len(config.datasets))

    # Load every corpus once up front: it fails fast on a bad path and lets the
    # pair tables be written before any GPU time is spent.
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
    for model_spec in config.enabled_models:
        encoder: Optional[TextEncoder] = None
        cache = EmbeddingCache(
            cache_dir=config.runtime.cache_dir,
            model_key=model_spec.key,
            fingerprint={**model_spec.fingerprint(), "resolved_dtype": dtype},
            enabled=config.runtime.use_cache,
        )
        try:
            for name, dataset in datasets.items():
                output_path = paths.prediction_file(model_spec.key, name)
                if config.resume and output_path.exists():
                    _log.info("[%s | %s] resume: reusing %s", model_spec.key, name, output_path)
                    predictions = pd.read_csv(output_path)
                else:
                    if encoder is None:
                        encoder = build_encoder(model_spec, config.runtime)
                        manifest.record_model(model_spec.key, model_spec.model_id, encoder.describe())
                    _log.info("[%s | %s] %d pairs, %d unique texts",
                              model_spec.key, name, len(dataset), len(dataset.unique_texts()))
                    started = time.time()
                    predictions = evaluate_model_on_dataset(
                        encoder, model_spec, dataset, cache, config.runtime.flush_every
                    )
                    elapsed = time.time() - started
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    predictions.to_csv(output_path, index=False)
                    manifest.record_run({
                        "model": model_spec.key,
                        "dataset": name,
                        "n_pairs": len(dataset),
                        "n_unique_texts": len(dataset.unique_texts()),
                        "runtime_seconds": round(elapsed, 2),
                        "pairs_per_second": round(len(dataset) / elapsed, 1) if elapsed else None,
                        "cache": cache.stats.to_dict(),
                        "n_truncated_pairs": int(predictions["pair_truncated"].sum()),
                        "predictions_file": str(output_path),
                    })
                    _log.info("[%s | %s] done in %.1fs -> %s",
                              model_spec.key, name, elapsed, output_path)

                all_metrics.extend(metric_rows(predictions, name, model_spec.key, config))
        finally:
            cache.flush()
            if encoder is not None:
                encoder.close()

    manifest.finish()
    return pd.DataFrame(all_metrics)
