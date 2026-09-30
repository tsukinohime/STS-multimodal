"""Command-line entry point.

    python -m sts.cli run      --config configs/text_only.yaml
    python -m sts.cli judge    --config configs/llm_judge.yaml
    python -m sts.cli validate --config configs/smoke.yaml
    python -m sts.cli align    --config configs/text_only.yaml \
                               --model-a jina-v5-text-small --model-b jina-v5-omni-small
    python -m sts.cli report   --config configs/text_only.yaml
    python -m sts.cli breakdown --config configs/llm_judge.yaml --datasets sick-all cxc-val \
                                --baseline outputs/text_only
    python -m sts.cli datasets

Every run is fully described by its YAML file; the flags below only exist to
override the machine-dependent parts (device, batch size) and to cut a run down
for a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .config import ExperimentConfig, get_logger, load_config
from .data import AVAILABLE_DATASETS
from .report import add_macro_average, print_console_table, write_metrics, write_summary

_log = get_logger()


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", "-c", required=True, type=Path, help="experiment YAML")
    parser.add_argument("--device", default=None, help="override runtime.device (auto|cuda|mps|cpu)")
    parser.add_argument("--dtype", default=None, help="override runtime.dtype")
    parser.add_argument("--batch-size", type=int, default=None, help="override runtime.batch_size")
    parser.add_argument("--limit", type=int, default=None, help="use only N pairs per dataset")
    parser.add_argument("--models", nargs="+", default=None, help="run only these model / judge keys")
    parser.add_argument("--datasets", nargs="+", default=None, help="run only these dataset names")
    parser.add_argument("--output-dir", type=Path, default=None, help="override experiment.output_dir")
    parser.add_argument("--data-root", type=Path, default=None, help="override data.root")
    parser.add_argument("--cache-dir", type=Path, default=None, help="override runtime.cache_dir")
    parser.add_argument("--run-tag", default=None,
                        help="suffix for manifest/metrics/summary filenames; one per parallel job")


def _config_from(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(args.config, overrides={
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "models": args.models,
        "datasets": args.datasets,
        "output_dir": args.output_dir,
        "data_root": args.data_root,
        "cache_dir": args.cache_dir,
        "run_tag": args.run_tag,
    })


def _load_manifest(config: ExperimentConfig) -> dict:
    """This run's manifest, or a merge of the per-model manifests.

    On TSUBAME each model runs as its own job and writes ``manifest__<key>.json``.
    A later ``report`` with no tag stitches them into one record.
    """
    tagged = config.tagged("manifest", ".json")
    if tagged.exists():
        return json.loads(tagged.read_text(encoding="utf-8"))

    parts = sorted(config.output_dir.glob("manifest__*.json"))
    if not parts:
        return {}
    merged = json.loads(parts[0].read_text(encoding="utf-8"))
    merged["merged_from"] = [p.name for p in parts]
    for part in parts[1:]:
        payload = json.loads(part.read_text(encoding="utf-8"))
        merged["models"].update(payload.get("models", {}))
        merged["datasets"].update(payload.get("datasets", {}))
        merged["runs"].extend(payload.get("runs", []))
        merged["total_runtime_seconds"] = round(
            (merged.get("total_runtime_seconds") or 0) + (payload.get("total_runtime_seconds") or 0), 2
        )
    return merged


def _finalise(metrics: pd.DataFrame, config: ExperimentConfig, method: str = "embedding") -> pd.DataFrame:
    metrics = add_macro_average(metrics, config.metrics.macro_average_over)
    write_metrics(metrics, config)
    write_summary(metrics, config, _load_manifest(config), method=method)
    print_console_table(metrics)
    return metrics


def _apply_run_flags(config: ExperimentConfig, args: argparse.Namespace) -> None:
    if args.no_resume:
        config.resume = False
    if args.no_cache:
        config.runtime.use_cache = False
    if args.bootstrap is not None:
        config.metrics.bootstrap = args.bootstrap


def command_run(args: argparse.Namespace) -> int:
    from .pipeline import run_experiment

    config = _config_from(args)
    _apply_run_flags(config, args)
    metrics = run_experiment(config)
    if metrics.empty:
        _log.error("No results produced.")
        return 1
    _finalise(metrics, config, method="embedding")
    return 0


def command_judge(args: argparse.Namespace) -> int:
    """LLM-as-judge: same datasets and outputs, a prompted LLM instead of an encoder."""
    from .judge.pipeline import run_judge_experiment

    config = _config_from(args)
    _apply_run_flags(config, args)
    if not config.enabled_judges:
        _log.error("No enabled judges in %s (add a `judges:` list or check --models).", config.source_path)
        return 1
    metrics = run_judge_experiment(config)
    if metrics.empty:
        _log.error("No results produced.")
        return 1
    _finalise(metrics, config, method="judge")
    return 0


def command_report(args: argparse.Namespace) -> int:
    """Rebuild metrics and the summary from prediction CSVs already on disk.

    Works for both experiment types: the prediction column is detected per
    file, and the summary header follows whichever kind of run was found.
    """
    from .pipeline import metric_rows, prediction_column_of

    config = _config_from(args)
    keys = [m.key for m in config.enabled_models] + [j.key for j in config.enabled_judges]
    rows: List[dict] = []
    saw_judge = False
    for key in keys:
        for name in config.datasets:
            path = config.output_dir / "predictions" / key / f"{name}.csv"
            if not path.exists():
                _log.warning("missing %s", path)
                continue
            frame = pd.read_csv(path)
            column = prediction_column_of(frame)
            if column == "expected_score":
                saw_judge = True
                n_all = len(frame)
                frame = frame[frame[column].notna()]
                extra = {"n_degenerate": n_all - len(frame)}
            else:
                extra = {}
            for row in metric_rows(frame, name, key, config, prediction_column=column):
                row.update(extra)
                rows.append(row)
    if not rows:
        _log.error("No prediction files found under %s", config.output_dir / "predictions")
        return 1
    _finalise(pd.DataFrame(rows), config, method="judge" if saw_judge else "embedding")
    return 0


def command_breakdown(args: argparse.Namespace) -> int:
    """Results split by SICK NLI label or dataset domain, from prediction CSVs already on disk."""
    from .breakdown import run_breakdown

    config = _config_from(args)
    written = run_breakdown(
        config, config.datasets,
        baselines=args.baseline or [],
        baseline_models=args.baseline_models,
        label_column=args.label_column,
        per_label=args.worst,
    )
    return 0 if written else 1


def command_validate(args: argparse.Namespace) -> int:
    from .diagnostics import print_validation, run_validation, write_validation_report

    config = _config_from(args)
    results = run_validation(config, encoder_checks=not args.data_only, sample=args.sample)
    write_validation_report(results, config.tagged("validation", ".json"))
    return 0 if print_validation(results) else 1


def command_align(args: argparse.Namespace) -> int:
    from .diagnostics import cross_model_alignment

    config = _config_from(args)
    report = cross_model_alignment(
        config, args.model_a, args.model_b, args.dataset, sample=args.sample
    )
    path = config.output_dir / f"alignment_{args.model_a}__{args.model_b}__{args.dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    cosine = report["same_text_cosine"]
    print(f"\nCross-model alignment: {report['model_a']} vs {report['model_b']}")
    print(f"  dataset                     {report['dataset']} ({report['n_pairs']} pairs, dim {report['dim']})")
    print(f"  cosine on the SAME text     mean {cosine['mean']:.4f} | median {cosine['median']:.4f} "
          f"| p05 {cosine['p05']:.4f} | min {cosine['min']:.4f}")
    print(f"  Spearman {report['model_a']:<18}  {report['spearman_a']:.4f}")
    print(f"  Spearman {report['model_b']:<18}  {report['spearman_b']:.4f}")
    print(f"  Spearman cross-encoded      {report['spearman_cross_encoded']:.4f} "
          f"(penalty {report['cross_encoding_penalty_vs_best']:+.4f})")
    print(f"  prediction agreement        {report['prediction_agreement_spearman']:.4f}")
    print(f"\n  -> {path}\n")
    return 0


def command_cache(args: argparse.Namespace) -> int:
    """Show, and optionally compact, this config's embedding caches."""
    from .cache import EmbeddingCache
    from .config import resolve_device, resolve_dtype

    config = _config_from(args)
    device = resolve_device(config.runtime.device)

    total_files = total_bytes = 0
    for model_spec in config.enabled_models:
        dtype = resolve_dtype(model_spec.dtype or config.runtime.dtype, device)
        cache = EmbeddingCache(
            cache_dir=config.runtime.cache_dir,
            model_key=model_spec.key,
            fingerprint={**model_spec.fingerprint(), "resolved_dtype": dtype},
            enabled=True,
        )
        if args.compact:
            result = cache.compact()
            stat = result["after"]
            note = (f"  ({result['before']['n_shards']} -> {stat['n_shards']} shards)"
                    if result["merged"] else "  (already compact)")
        else:
            stat = cache.stat()
            note = ""
        total_files += stat["n_shards"] + 1  # + meta.json
        total_bytes += stat["bytes"]
        print(f"\n{model_spec.key}  [dtype={dtype}]")
        print(f"  {stat['directory']}")
        print(f"  {stat['n_vectors']:,} vectors | {stat['n_shards']} shard(s) "
              f"| {stat['bytes'] / 1024 ** 2:.1f} MB "
              f"| mean {stat['mean_shard_bytes'] / 1024 ** 2:.1f} MB/shard{note}")

    print(f"\nTotal: {total_files} file(s), {total_bytes / 1024 ** 2:.1f} MB\n")
    return 0


def command_datasets(_: argparse.Namespace) -> int:
    print("\nAvailable datasets:\n")
    for name, description in sorted(AVAILABLE_DATASETS.items()):
        print(f"  {name:<12} {description}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sts", description="Text-only zero-shot STS evaluation (frozen encoders, cosine similarity)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the experiment end to end")
    _add_common(run_parser)
    run_parser.add_argument("--no-resume", action="store_true", help="recompute even if predictions exist")
    run_parser.add_argument("--no-cache", action="store_true", help="bypass the embedding cache")
    run_parser.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples for the CI")
    run_parser.set_defaults(func=command_run)

    judge_parser = subparsers.add_parser(
        "judge", help="LLM-as-judge STS: prompt a chat model, take the expected digit"
    )
    _add_common(judge_parser)
    judge_parser.add_argument("--no-resume", action="store_true", help="recompute even if predictions exist")
    judge_parser.add_argument("--no-cache", action="store_true", help="bypass the judgment cache")
    judge_parser.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples for the CI")
    judge_parser.set_defaults(func=command_judge)

    report_parser = subparsers.add_parser("report", help="rebuild metrics/summary from existing predictions")
    _add_common(report_parser)
    report_parser.set_defaults(func=command_report)

    breakdown_parser = subparsers.add_parser(
        "breakdown", help="results split by SICK NLI label / CxC sampling_method, from existing "
                          "predictions (no model loaded)"
    )
    _add_common(breakdown_parser)
    breakdown_parser.add_argument("--baseline", type=Path, action="append", default=None,
                                  help="another experiment's output dir; its predictions/<model>/ "
                                       "files are compared too (repeatable)")
    breakdown_parser.add_argument("--baseline-models", nargs="+", default=None,
                                  help="only these model keys from the --baseline dirs")
    breakdown_parser.add_argument("--label-column", default="entailment_label",
                                  choices=["entailment_label", "entailment_AB", "entailment_BA"],
                                  help="SICK column to stratify by; other datasets use their domain")
    breakdown_parser.add_argument("--worst", type=int, default=10,
                                  help="pairs per (system, label) in the worst-pairs CSV")
    breakdown_parser.set_defaults(func=command_breakdown)

    validate_parser = subparsers.add_parser("validate", help="run the correctness checks")
    _add_common(validate_parser)
    validate_parser.add_argument("--data-only", action="store_true", help="skip checks that load a model")
    validate_parser.add_argument("--sample", type=int, default=16, help="sentences per encoder check")
    validate_parser.set_defaults(func=command_validate)

    align_parser = subparsers.add_parser(
        "align", help="measure whether two encoders share one embedding space"
    )
    _add_common(align_parser)
    align_parser.add_argument("--model-a", required=True, help="model key from the config")
    align_parser.add_argument("--model-b", required=True, help="model key from the config")
    align_parser.add_argument("--dataset", default="sick-test", help="dataset to measure on")
    align_parser.add_argument("--sample", type=int, default=512, help="pairs to sample")
    align_parser.set_defaults(func=command_align)

    cache_parser = subparsers.add_parser(
        "cache", help="show embedding-cache size, or merge its shards into one file"
    )
    _add_common(cache_parser)
    cache_parser.add_argument("--compact", action="store_true",
                              help="merge all shards into one file (fewer inodes on a cluster)")
    cache_parser.set_defaults(func=command_cache)

    datasets_parser = subparsers.add_parser("datasets", help="list the registered datasets")
    datasets_parser.set_defaults(func=command_datasets)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
