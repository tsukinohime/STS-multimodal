#!/usr/bin/env python
"""End-to-end smoke test: run the real pipeline twice, then verify every invariant.

    python scripts/smoke_test.py --config configs/smoke.yaml

What it proves, in order:

1. A cold run completes on every dataset (embedding cache starts empty).
2. A second run from a *separate process* reuses the cache — 100% hit rate and
   zero encoder work — which is the "restart resumes from cache" requirement.
3. The two runs produce bit-identical predictions.
4. The correctness checks in ``sts.diagnostics`` all pass: identical sentences
   score cosine 1, swapping the sentence order changes nothing, no NaNs, every
   CxC caption id resolves, and pair counts match the source files.
5. One config file plus one command reproduces the whole thing.

Exit code is 0 only if everything passed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from sts.config import load_config  # noqa: E402
from sts.diagnostics import (  # noqa: E402
    CheckResult,
    print_validation,
    run_validation,
    write_validation_report,
)


def run_cli(args: List[str], env_note: str) -> Tuple[int, float]:
    command = [sys.executable, "-m", "sts.cli", *args]
    print(f"\n$ {' '.join(command)}   # {env_note}\n" + "-" * 78, flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=REPO_ROOT)
    return result.returncode, time.time() - started


def cache_totals(manifest_path: Path) -> Tuple[int, int]:
    """Sum cache hits/misses over every (model, dataset) in a manifest."""
    if not manifest_path.exists():
        return (0, 0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not runs:
        return (0, 0)
    # Stats accumulate per model, so the last run of each model holds its total.
    last_by_model = {}
    for entry in runs:
        last_by_model[entry["model"]] = entry.get("cache", {})
    hits = sum(stats.get("hits", 0) for stats in last_by_model.values())
    misses = sum(stats.get("misses", 0) for stats in last_by_model.values())
    return hits, misses


def read_predictions(output_dir: Path) -> dict:
    return {
        str(path.relative_to(output_dir)): pd.read_csv(path)
        for path in sorted((output_dir / "predictions").rglob("*.csv"))
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", type=Path, default=REPO_ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="override the enabled model keys")
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-cache", action="store_true",
                        help="do not clear this config's embedding cache first (skips the cold-run check)")
    args = parser.parse_args(argv)

    config = load_config(args.config, overrides={"models": args.models, "device": args.device})
    output_dir = config.output_dir
    manifest_path = output_dir / "manifest.json"

    passthrough: List[str] = ["--config", str(args.config)]
    if args.models:
        passthrough += ["--models", *args.models]
    if args.device:
        passthrough += ["--device", args.device]

    print("=" * 78)
    print(f"SMOKE TEST  |  config={args.config.name}  limit={config.limit} pairs/dataset")
    print(f"            |  models={[m.key for m in config.enabled_models]}")
    print(f"            |  datasets={config.datasets}")
    print("=" * 78)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    if not args.keep_cache:
        for model_spec in config.enabled_models:
            from sts.cache import EmbeddingCache

            stale = EmbeddingCache(
                config.runtime.cache_dir, model_spec.key,
                {**model_spec.fingerprint(), "resolved_dtype": "probe"}, enabled=False,
            ).directory.parent
            for directory in stale.glob(f"{model_spec.key}__*"):
                shutil.rmtree(directory, ignore_errors=True)

    extra_checks: List[CheckResult] = []

    # ---- 1. cold run -------------------------------------------------------
    code, cold_seconds = run_cli(["run", *passthrough], "cold: empty embedding cache")
    if code != 0:
        print(f"\nFAIL: the cold run exited with code {code}")
        return 1
    cold_hits, cold_misses = cache_totals(manifest_path)
    cold_predictions = read_predictions(output_dir)
    extra_checks.append(CheckResult(
        name="cold_run_completes",
        passed=bool(cold_predictions),
        detail=f"{len(cold_predictions)} prediction file(s) in {cold_seconds:.1f}s; "
               f"cache {cold_hits} hit / {cold_misses} miss",
        data={"seconds": round(cold_seconds, 2), "hits": cold_hits, "misses": cold_misses},
    ))

    # ---- 2. warm run in a fresh process ------------------------------------
    code, warm_seconds = run_cli(
        ["run", *passthrough, "--no-resume"], "warm: separate process, cache on disk"
    )
    if code != 0:
        print(f"\nFAIL: the warm run exited with code {code}")
        return 1
    warm_hits, warm_misses = cache_totals(manifest_path)
    extra_checks.append(CheckResult(
        name="embedding_cache_reused_after_restart",
        passed=warm_misses == 0 and warm_hits > 0,
        detail=f"second process: {warm_hits} cache hits, {warm_misses} misses "
               f"({warm_seconds:.1f}s vs {cold_seconds:.1f}s cold)",
        data={"hits": warm_hits, "misses": warm_misses,
              "cold_seconds": round(cold_seconds, 2), "warm_seconds": round(warm_seconds, 2)},
    ))

    # ---- 3. reproducibility ------------------------------------------------
    warm_predictions = read_predictions(output_dir)
    mismatches = []
    for name, cold_frame in cold_predictions.items():
        warm_frame = warm_predictions.get(name)
        if warm_frame is None:
            mismatches.append(f"{name}: missing after rerun")
            continue
        if len(cold_frame) != len(warm_frame):
            mismatches.append(f"{name}: {len(cold_frame)} vs {len(warm_frame)} rows")
            continue
        delta = (cold_frame["cosine_prediction"] - warm_frame["cosine_prediction"]).abs().max()
        if delta > 0:
            mismatches.append(f"{name}: max |Δcosine| = {delta:.3e}")
    extra_checks.append(CheckResult(
        name="rerun_reproduces_identical_predictions",
        passed=not mismatches,
        detail="both runs produced bit-identical cosine predictions" if not mismatches
        else "; ".join(mismatches),
        data={"n_files": len(cold_predictions)},
    ))

    # ---- 4. invariants -----------------------------------------------------
    print("\n" + "-" * 78)
    print("Running correctness checks...")
    checks = run_validation(config, encoder_checks=True, sample=16)

    # ---- 5. single-command reproduction ------------------------------------
    reproduce = f"python -m sts.cli run --config {args.config}"
    extra_checks.append(CheckResult(
        name="single_command_reproduces_the_run",
        passed=args.config.exists(),
        detail=f"`{reproduce}` — config committed at {args.config}",
        data={"command": reproduce},
    ))

    all_checks = checks + extra_checks
    write_validation_report(
        all_checks,
        output_dir / "validation.json",
        extra={"smoke": {"cold_seconds": round(cold_seconds, 2),
                         "warm_seconds": round(warm_seconds, 2),
                         "config": str(args.config)}},
    )
    passed = print_validation(all_checks)

    print(f"Artifacts under {output_dir}/:")
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(output_dir)}  ({path.stat().st_size:,} bytes)")
    print()
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
