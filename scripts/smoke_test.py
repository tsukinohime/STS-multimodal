#!/usr/bin/env python
"""End-to-end smoke test: run the real pipeline twice, then verify every invariant.

    python scripts/smoke_test.py --config configs/smoke.yaml                 # encoders
    python scripts/smoke_test.py --config configs/llm_judge_smoke.yaml --judge   # LLM judge
    python scripts/smoke_test.py --config configs/d1_smoke.yaml --d1          # D1: images

What it proves, in order:

1. A cold run completes on every dataset (cache starts empty).
2. A second run from a *separate process* reuses the cache — 100% hit rate and
   zero model work — which is the "restart resumes from cache" requirement.
3. The two runs produce bit-identical predictions.
4. The correctness checks in ``sts.diagnostics`` all pass: no NaNs, every CxC
   caption id resolves, pair counts match the source files, and — for encoders —
   identical sentences score cosine 1 and swapping the order changes nothing.
   For a judge the encoder probes are replaced by prediction-level invariants:
   every row carries a traceable prompt hash, digit probabilities are
   normalised, the model actually answered with digits, and the expectation
   breaks the ties an argmax answer would leave.
5. One config file plus one command reproduces the whole thing.

With ``--d1`` each run is three stages — ``imagine``, then ``run`` and ``judge``
on the generated images — and three more checks apply: every sentence has an
image, a second ``imagine`` generates nothing, and no sentence text reaches an
image-only prompt while both images do.

Exit code is 0 only if everything passed.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sts.config import load_config  # noqa: E402
from sts.diagnostics import (  # noqa: E402
    CheckResult,
    print_validation,
    run_validation,
    write_validation_report,
)
from sts.pipeline import prediction_column_of  # noqa: E402


def run_cli(args: List[str], env_note: str) -> Tuple[int, float]:
    command = [sys.executable, "-m", "sts.cli", *args]
    print(f"\n$ {' '.join(command)}   # {env_note}\n" + "-" * 78, flush=True)
    started = time.time()
    result = subprocess.run(command, cwd=REPO_ROOT)
    return result.returncode, time.time() - started


def cache_totals(manifest_path: Path, by_family: bool = False) -> Tuple[int, int]:
    """Sum cache hits/misses over every cache object a manifest reports on.

    Stats accumulate per cache object, so the *last* run entry that used each
    object holds its total. Encoders keep one cache per model; the judge keeps
    one per (judge, dataset family) because the prompt differs per family.
    """
    if not manifest_path.exists():
        return (0, 0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs", [])
    if not runs:
        return (0, 0)
    last_by_cache = {}
    for entry in runs:
        key = (entry["model"], entry["dataset"].split("-")[0] if by_family else "")
        last_by_cache[key] = entry.get("cache", {})
    hits = sum(stats.get("hits", 0) for stats in last_by_cache.values())
    misses = sum(stats.get("misses", 0) for stats in last_by_cache.values())
    return hits, misses


def stage_cache_totals(output_dir: Path, d1: bool, judge: bool) -> Tuple[int, int]:
    """Cache hits/misses of this run; a D1 run has one manifest per scoring stage."""
    if not d1:
        return cache_totals(output_dir / "manifest.json", by_family=judge)
    hits_e, misses_e = cache_totals(output_dir / "manifest__embedding.json")
    hits_j, misses_j = cache_totals(output_dir / "manifest__judge.json", by_family=True)
    return hits_e + hits_j, misses_e + misses_j


def read_predictions(output_dir: Path) -> dict:
    return {
        str(path.relative_to(output_dir)): pd.read_csv(path)
        for path in sorted((output_dir / "predictions").rglob("*.csv"))
    }


def judge_checks(output_dir: Path) -> List[CheckResult]:
    """Invariants specific to LLM-judge predictions (there is no encoder to probe)."""
    frames = {name: f for name, f in read_predictions(output_dir).items() if "expected_score" in f}
    results: List[CheckResult] = []

    # 1. Every row carries a prompt hash, one per file, consistent within a
    #    dataset family, and the rendered prompt was dumped for tracing.
    bad_hash, missing_dump = [], []
    hashes_by_family: dict = {}
    for name, frame in frames.items():
        if "prompt_hash" not in frame or frame["prompt_hash"].isna().any() or frame["prompt_hash"].nunique() != 1:
            bad_hash.append(name)
            continue
        digest = str(frame["prompt_hash"].iloc[0])
        family = str(frame["dataset"].iloc[0]).split("-")[0]
        hashes_by_family.setdefault(family, set()).add(digest)
        if not list((output_dir / "prompts").glob(f"{family}__*__{digest[:12]}.md")):
            missing_dump.append(name)
    inconsistent = {f: sorted(h) for f, h in hashes_by_family.items() if len(h) > 1}
    problems = []
    if bad_hash:
        problems.append(f"missing/mixed hash in {bad_hash}")
    if missing_dump:
        problems.append(f"no prompt dump for {missing_dump}")
    if inconsistent:
        problems.append(f"family with several hashes {inconsistent}")
    results.append(CheckResult(
        name="prompt_hash_recorded_and_traceable",
        passed=not problems,
        detail=(f"{len(frames)} file(s), one hash per dataset family "
                f"{ {f: next(iter(h))[:12] for f, h in hashes_by_family.items()} }, "
                "each dumped under outputs/prompts/") if not problems else "; ".join(problems),
    ))

    # 2. Digit probabilities are a distribution and the expectation lies on the scale.
    worst_sum, out_of_scale = 0.0, 0
    for frame in frames.values():
        probs = frame[[c for c in frame.columns if c.startswith("p_")]]
        finite = frame["expected_score"].notna()
        if finite.any():
            worst_sum = max(worst_sum, float((probs[finite].sum(axis=1) - 1.0).abs().max()))
            lo, hi = float(frame["scale_min"].iloc[0]), float(frame["scale_max"].iloc[0])
            e = frame.loc[finite, "expected_score"]
            out_of_scale += int(((e < lo - 1e-6) | (e > hi + 1e-6)).sum())
    results.append(CheckResult(
        name="digit_probabilities_normalised_and_expectation_on_scale",
        passed=worst_sum < 1e-4 and out_of_scale == 0,
        detail=f"max |Σp - 1| = {worst_sum:.2e}; {out_of_scale} expectation(s) outside the scale",
    ))

    # 3. The judge actually answered with a digit (thinking off, prompt followed).
    masses = np.concatenate([f["valid_mass"].to_numpy() for f in frames.values()])
    degenerate = sum(int(f["expected_score"].isna().sum()) for f in frames.values())
    results.append(CheckResult(
        name="judge_answers_with_a_digit",
        passed=float(np.median(masses)) >= 0.5 and degenerate == 0,
        detail=f"digit-token mass median {np.median(masses):.3f}, min {masses.min():.3f}; "
               f"{degenerate} degenerate row(s)",
    ))

    # 4. The expectation breaks the ties an argmax answer leaves.
    n_exp = sum(f["expected_score"].round(6).nunique() for f in frames.values())
    n_arg = sum(f["argmax_score"].nunique() for f in frames.values())
    results.append(CheckResult(
        name="expectation_breaks_argmax_ties",
        passed=n_exp > n_arg,
        detail=f"distinct values summed over files: expectation {n_exp} vs argmax {n_arg}",
    ))
    return results


def d1_checks(config, output_dir: Path, warm_generated: Optional[int]) -> List[CheckResult]:
    """Invariants of the image conditions: complete store, stable store, image-only prompts."""
    from sts.images.pipeline import load_datasets
    from sts.images.store import open_image_store, visual_tokens
    from sts.judge.prompts import PROMPT_SETS

    results: List[CheckResult] = []
    store = open_image_store(config)
    texts = list(dict.fromkeys(t for d in load_datasets(config).values() for t in d.unique_texts()))
    missing = [t for t in texts if t not in store]
    results.append(CheckResult(
        name="every_sentence_has_a_generated_image",
        passed=not missing,
        detail=f"{len(texts) - len(missing)}/{len(texts)} sentences have an image in {store.directory.name}",
    ))
    results.append(CheckResult(
        name="image_store_reused_after_restart",
        passed=warm_generated == 0,
        detail=f"second `imagine` generated {warm_generated} image(s)",
    ))

    # No sentence text may reach an image-only prompt: by construction (the
    # specs render without the sentences) and in every prompt actually dumped.
    leaks = []
    image_judges = [j for j in config.enabled_judges if j.input == "image"]
    for prompt_set in {j.prompt_set for j in image_judges}:
        for family, spec in PROMPT_SETS[prompt_set].items():
            if "SENTINEL" in json.dumps(spec.messages("SENTINEL_A", "SENTINEL_B")):
                leaks.append(f"{prompt_set}/{family} spec")
    dumps = sorted((output_dir / "prompts").glob("*__image-*.md"))
    for dump in dumps:
        text = dump.read_text(encoding="utf-8")
        pair_id = re.search(r"pair `([^`]+)`", text).group(1)
        pairs = pd.read_csv(output_dir / "pairs" / f"{pair_id.rsplit('-', 1)[0]}.csv").set_index("pair_id")
        rendered = text.split("exactly as tokenised", 1)[1]
        if any(str(pairs.loc[pair_id, column]) in rendered for column in ("sentence1", "sentence2")):
            leaks.append(dump.name)
    results.append(CheckResult(
        name="image_prompts_contain_no_sentence_text",
        passed=not leaks and bool(dumps),
        detail=f"{len(dumps)} dumped image prompt(s) and every image spec checked" if not leaks
        else f"sentence text found in {leaks}",
    ))

    # Both images reached the judge: each prompt is at least two images long.
    short = []
    for judge in image_judges:
        per_image = visual_tokens(store.fingerprint["width"], store.fingerprint["height"], judge.image_size)
        for path in sorted((output_dir / "predictions" / judge.key).glob("*.csv")):
            shortest = int(pd.read_csv(path)["prompt_tokens"].min())
            if shortest < 2 * per_image:
                short.append(f"{path.name}: {shortest} < 2 x {per_image}")
    results.append(CheckResult(
        name="judge_prompts_carry_two_images",
        passed=not short and bool(image_judges),
        detail="every image-judge prompt holds at least two images' worth of image tokens" if not short
        else "; ".join(short),
    ))
    store.close()
    return results


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", "-c", type=Path, default=REPO_ROOT / "configs" / "smoke.yaml")
    parser.add_argument("--models", nargs="+", default=None, help="override the enabled model keys")
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="override data.root (e.g. the WORK disk on a cluster)")
    parser.add_argument("--keep-cache", action="store_true",
                        help="do not clear this config's embedding cache first (skips the cold-run check)")
    parser.add_argument("--judge", action="store_true",
                        help="smoke-test the LLM-as-judge pipeline (`sts.cli judge`) instead of the encoder one")
    parser.add_argument("--d1", action="store_true",
                        help="smoke-test condition D1: `imagine`, then `run` and `judge` on the generated images")
    args = parser.parse_args(argv)
    if args.d1 and args.judge:
        parser.error("--d1 already covers the judge; pass one of --judge / --d1")
    stages = ["imagine", "run", "judge"] if args.d1 else ["judge" if args.judge else "run"]

    config = load_config(args.config, overrides={
        "models": args.models, "device": args.device, "data_root": args.data_root,
    })
    output_dir = config.output_dir

    passthrough: List[str] = ["--config", str(args.config)]
    if args.models:
        passthrough += ["--models", *args.models]
    if args.device:
        passthrough += ["--device", args.device]
    if args.data_root:
        passthrough += ["--data-root", str(args.data_root)]

    print("=" * 78)
    print(f"SMOKE TEST  |  config={args.config.name}  limit={config.limit} pairs/dataset")
    keys = [m.key for m in config.enabled_models] + [j.key for j in config.enabled_judges]
    print(f"            |  {'judges' if args.judge else 'models'}={keys}")
    print(f"            |  datasets={config.datasets}")
    print("=" * 78)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    if not args.keep_cache:
        # Cache directories are named <key>__<fingerprint> (encoders) or
        # <key>__<family>__<fingerprint> (judges); both match <key>__*.
        for key in keys:
            for directory in config.runtime.cache_dir.glob(f"{key}__*"):
                shutil.rmtree(directory, ignore_errors=True)
        generator = config.images.generator
        if args.d1 and generator is not None and generator.backend == "render-text":
            # Only the model-free stand-in's images are cheap enough to throw
            # away; a real generator's store is shared with the full run.
            from sts.images.store import open_image_store

            shutil.rmtree(open_image_store(config).directory, ignore_errors=True)

    def run_stages(extra: List[str], note: str) -> Tuple[int, float]:
        total = 0.0
        for stage in stages:
            flags = extra if stage != "imagine" else []
            code, seconds = run_cli([stage, *passthrough, *flags], note)
            total += seconds
            if code != 0:
                return code, total
        return 0, total

    def generated_now() -> Optional[int]:
        path = output_dir / "imagine.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["runs"][-1]["n_generated"]

    extra_checks: List[CheckResult] = []

    # ---- 1. cold run -------------------------------------------------------
    code, cold_seconds = run_stages([], "cold: empty cache")
    if code != 0:
        print(f"\nFAIL: the cold run exited with code {code}")
        return 1
    cold_hits, cold_misses = stage_cache_totals(output_dir, args.d1, args.judge)
    cold_predictions = read_predictions(output_dir)
    extra_checks.append(CheckResult(
        name="cold_run_completes",
        passed=bool(cold_predictions),
        detail=f"{len(cold_predictions)} prediction file(s) in {cold_seconds:.1f}s; "
               f"cache {cold_hits} hit / {cold_misses} miss",
        data={"seconds": round(cold_seconds, 2), "hits": cold_hits, "misses": cold_misses},
    ))

    # ---- 2. warm run in a fresh process ------------------------------------
    code, warm_seconds = run_stages(["--no-resume"], "warm: separate process, cache on disk")
    if code != 0:
        print(f"\nFAIL: the warm run exited with code {code}")
        return 1
    warm_generated = generated_now()
    warm_hits, warm_misses = stage_cache_totals(output_dir, args.d1, args.judge)
    extra_checks.append(CheckResult(
        name="cache_reused_after_restart",
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
        column = prediction_column_of(cold_frame)
        # A NaN (degenerate judge answer) on both sides counts as agreement.
        delta = (cold_frame[column].fillna(-999.0) - warm_frame[column].fillna(-999.0)).abs().max()
        if delta > 0:
            mismatches.append(f"{name}: max |Δ{column}| = {delta:.3e}")
    extra_checks.append(CheckResult(
        name="rerun_reproduces_identical_predictions",
        passed=not mismatches,
        detail="both runs produced bit-identical predictions" if not mismatches
        else "; ".join(mismatches),
        data={"n_files": len(cold_predictions)},
    ))

    # ---- 4. invariants -----------------------------------------------------
    print("\n" + "-" * 78)
    print("Running correctness checks...")
    # The encoder checks (cos(x,x)=1, order invariance) are properties of an
    # embedding space; a judge is checked through its predictions instead.
    checks = run_validation(config, encoder_checks=not args.judge, sample=16)
    if args.judge or args.d1:
        checks.extend(judge_checks(output_dir))
    if args.d1:
        checks.extend(d1_checks(config, output_dir, warm_generated))

    # ---- 5. single-command reproduction ------------------------------------
    reproduce = " && ".join(f"python -m sts.cli {stage} --config {args.config}" for stage in stages)
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
