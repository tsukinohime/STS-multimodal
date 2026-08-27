"""Command-line entry point.

    python -m sts.cli run      --config configs/text_only.yaml
    python -m sts.cli validate --config configs/smoke.yaml
    python -m sts.cli align    --config configs/text_only.yaml \
                               --model-a jina-v5-text-small --model-b jina-v5-omni-small
    python -m sts.cli report   --config configs/text_only.yaml
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
    parser.add_argument("--models", nargs="+", default=None, help="run only these model keys")
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


def _finalise(metrics: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    metrics = add_macro_average(metrics, config.metrics.macro_average_over)
    write_metrics(metrics, config)
    write_summary(metrics, config, _load_manifest(config))
    print_console_table(metrics)
    return metrics


def command_run(args: argparse.Namespace) -> int:
    from .pipeline import run_experiment

    config = _config_from(args)
    if args.no_resume:
        config.resume = False
    if args.no_cache:
        config.runtime.use_cache = False
    if args.bootstrap is not None:
        config.metrics.bootstrap = args.bootstrap

    metrics = run_experiment(config)
    if metrics.empty:
        _log.error("No results produced.")
        return 1
    _finalise(metrics, config)
    return 0


def command_report(args: argparse.Namespace) -> int:
    """Rebuild metrics and the summary from prediction CSVs already on disk."""
    from .pipeline import metric_rows

    config = _config_from(args)
    rows: List[dict] = []
    for model_spec in config.enabled_models:
        for name in config.datasets:
            path = config.output_dir / "predictions" / model_spec.key / f"{name}.csv"
            if not path.exists():
                _log.warning("missing %s", path)
                continue
            rows.extend(metric_rows(pd.read_csv(path), name, model_spec.key, config))
    if not rows:
        _log.error("No prediction files found under %s", config.output_dir / "predictions")
        return 1
    _finalise(pd.DataFrame(rows), config)
    return 0


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

    report_parser = subparsers.add_parser("report", help="rebuild metrics/summary from existing predictions")
    _add_common(report_parser)
    report_parser.set_defaults(func=command_report)

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

    datasets_parser = subparsers.add_parser("datasets", help="list the registered datasets")
    datasets_parser.set_defaults(func=command_datasets)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
