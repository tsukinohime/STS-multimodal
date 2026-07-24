"""CLI entry point for text-only zero-shot STS evaluation.

Examples
--------
    # default encoder (all-mpnet-base-v2) on SICK-R test + CxC val/test
    python -m sts.run --datasets sick cxc-val cxc-test

    # quick smoke test on a subsample
    python -m sts.run --datasets sick cxc-val --limit 200

    # a different encoder
    python -m sts.run --datasets sick --encoder sentence-transformers/all-MiniLM-L6-v2

    # an E5 model (needs a prompt prefix)
    python -m sts.run --datasets sick --encoder intfloat/e5-base-v2 --prompt-prefix "query: "
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List

from .config import Config, get_logger
from .evaluate import available_datasets, run_evaluation

_log = get_logger()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        data_root=args.data_root or Config.data_root,
        encoder_name=args.encoder,
        encoder_backend=args.backend,
        batch_size=args.batch_size,
        device=args.device,
        prompt_prefix=args.prompt_prefix,
        max_seq_length=args.max_seq_length,
        hf_pooling=args.hf_pooling,
        sick_split=args.sick_split,
        limit=args.limit,
        seed=args.seed,
        output_dir=args.output_dir or Config.output_dir,
    )


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot text-only STS evaluation.")
    p.add_argument("--datasets", nargs="+", default=["sick", "cxc-val", "cxc-test"],
                   help=f"datasets to evaluate; available: {available_datasets()} "
                        f"(SICK split override via e.g. 'sick:all')")
    p.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2",
                   help="HF/SBERT model name")
    p.add_argument("--backend", default="sentence-transformers",
                   choices=["sentence-transformers", "hf-mean"],
                   help="encoder backend")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    p.add_argument("--prompt-prefix", default="", help="prepended to each text (e.g. 'query: ' for E5)")
    p.add_argument("--max-seq-length", type=int, default=None)
    p.add_argument("--hf-pooling", default="mean", choices=["mean", "cls"],
                   help="pooling for --backend hf-mean")
    p.add_argument("--sick-split", default="test", choices=["all", "train", "trial", "test"])
    p.add_argument("--limit", type=int, default=None, help="subsample N pairs per dataset (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kendall", action="store_true", help="also compute Kendall's tau")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--no-save", action="store_true", help="do not write a results JSON")
    return p.parse_args(argv)


def _print_table(results: List[dict]) -> None:
    header = f"{'dataset':<20}{'n':>8}{'spearman':>11}{'pearson':>10}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        n = "" if r.get("n") is None else str(r["n"])
        kendall = f"  kendall={r['kendall']:.4f}" if r.get("kendall") is not None else ""
        print(f"{r['dataset']:<20}{n:>8}{r['spearman']:>11.4f}{r['pearson']:>10.4f}{kendall}")
    print()


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)

    _log.info("Encoder: %s (%s) | device=%s", config.encoder_name, config.encoder_backend, config.device)
    results = run_evaluation(config, args.datasets, with_kendall=args.kendall)
    _print_table(results)

    if not args.no_save:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = config.encoder_name.replace("/", "__")
        out_path = config.output_dir / f"text_sts_{safe_model}_{stamp}.json"
        payload = {"config": config.to_dict(), "datasets": args.datasets, "results": results}
        out_path.write_text(json.dumps(payload, indent=2))
        _log.info("Saved results -> %s", out_path)


if __name__ == "__main__":
    main()
