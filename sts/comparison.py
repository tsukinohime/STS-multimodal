"""Pair-by-pair comparison of a judge run with the runs it controls for (``sts.cli compare``).

    python -m sts.cli compare --config configs/e2_mismatched.yaml      # CPU, no model loaded

Built for condition E2: the D2 judge, prompt unchanged, shown every pair with
another pair's images. The config's judge is set against the finished runs
under ``comparison:`` on the pairs it scored: ``matched``, the same judge and
prompt shown each pair's own images (D2), and optionally ``no_image``, the same
checkpoint on the sentence pair alone (B1). Per dataset:

* **Correlation with gold** of every system on the same pairs, and each paired
  difference (this run − matched, this run − no image, matched − no image) with
  a paired-bootstrap CI (both systems on the same resampled pairs).
* **Score shift**: the mean |difference| between two systems' scores on the
  judge's scale, and the Spearman between their scores. This run and
  ``matched`` render the same prompt text with different images, so their
  shift is the images' doing alone; a judge that does not look at its images
  gives both the same score, pair for pair.
* **Image following**, when the predictions name a donor (E2): the Spearman
  between this run − matched and donor gold − gold. Positive when a score moves
  toward the similarity of the pair the shown images were generated for.
  ``gold vs donor gold`` checks the randomisation and should be near 0.

Every reference must have predicted every pair the run scored (a full run
covers a smoke run's pairs). Pairs any system left degenerate are dropped from
all of them and counted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import ComparisonConfig, ExperimentConfig, FusionSource, get_logger
from .data import STSDataset, load_dataset
from .fusion import paired_bootstrap, read_source
from .manifest import Manifest
from .metrics import compute_correlations

_log = get_logger()

#: The run's own label in column names; the references go by their role.
RUN = "run"
#: Short names in the comparison tables, and what each stands for.
LABELS = {"no_image": "none", "matched": "own", RUN: "other"}
LEGEND = {
    RUN: "this run: each sentence pair shown another pair's generated images",
    "matched": "`comparison.matched`: each pair shown its own images",
    "no_image": "`comparison.no_image`: the sentence pair alone",
}
#: Paired differences, as (system, reference), in report order.
CONTRASTS = ((RUN, "matched"), (RUN, "no_image"), ("matched", "no_image"))


def _scale(frame: pd.DataFrame) -> Optional[Tuple[float, float]]:
    """The judge's answer scale, or None for a run without one (an encoder)."""
    if {"scale_min", "scale_max"} <= set(frame.columns):
        return float(frame["scale_min"].iloc[0]), float(frame["scale_max"].iloc[0])
    return None


def compare_dataset(
    key: str,
    dataset: STSDataset,
    run: pd.DataFrame,
    references: Dict[str, pd.DataFrame],
    comparison: ComparisonConfig,
    seed: int,
    confidence: float,
) -> Dict[str, object]:
    """One row of the comparison: the run against every reference on ``dataset``."""
    for role, frame in references.items():
        missing = run.index.difference(frame.index)
        if len(missing):
            raise ValueError(
                f"comparison.{role} has no prediction for {len(missing)} of the {len(run)} {dataset.name} pairs "
                f"{key} scored (e.g. {missing[0]}); compare with a run over the same pairs, or a full one"
            )
    scores = pd.DataFrame({RUN: run["score"], **{role: f.loc[run.index, "score"] for role, f in references.items()}})
    scored = scores.notna().all(axis=1).to_numpy()
    gold = run["gold_score"].to_numpy(np.float64)[scored]
    values = {name: scores[name].to_numpy(np.float64)[scored] for name in scores.columns}
    scales = {RUN: _scale(run), **{role: _scale(f) for role, f in references.items()}}

    run_scale = scales[RUN]
    row: Dict[str, object] = {
        "system": key, "dataset": dataset.name, "n": int(gold.size), "n_unscored": int((~scored).sum()),
        "scale_width": run_scale[1] - run_scale[0] if run_scale else float("nan"),
    }
    for name, value in values.items():
        correlations = compute_correlations(value, gold)
        row[f"{name}_spearman"], row[f"{name}_pearson"] = correlations.spearman, correlations.pearson

    contrasts = [(a, b) for a, b in CONTRASTS if a in values and b in values]
    for reference in dict.fromkeys(b for _, b in contrasts):
        others = {a: values[a] for a, b in contrasts if b == reference}
        intervals = (paired_bootstrap(gold, values[reference], others, comparison.bootstrap, seed, confidence)
                     if comparison.bootstrap else {})
        for other, value in others.items():
            name = f"{other}_vs_{reference}"
            for metric in ("spearman", "pearson"):
                row[f"{name}_{metric}"] = row[f"{other}_{metric}"] - row[f"{reference}_{metric}"]
                low, high = intervals.get(other, {}).get(metric, (np.nan, np.nan))
                row[f"{name}_{metric}_ci_low"], row[f"{name}_{metric}_ci_high"] = low, high
            same_scale = scales[other] is not None and scales[other] == scales[reference]
            row[f"{name}_agreement"] = compute_correlations(value, values[reference]).spearman
            row[f"{name}_mean_abs_shift"] = (float(np.mean(np.abs(value - values[reference])))
                                             if same_scale else float("nan"))

    if "donor_gold_score" in run.columns:
        donor_gold = run["donor_gold_score"].to_numpy(np.float64)[scored]
        row["gold_vs_donor_gold"] = compute_correlations(donor_gold, gold).spearman
        row["run_vs_donor_gold"] = compute_correlations(values[RUN], donor_gold).spearman
        row["image_following"] = (compute_correlations(values[RUN] - values["matched"], donor_gold - gold).spearman
                                  if "matched" in values else float("nan"))
    return row


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(value, digits: int = 4, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|",
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _delta(row: Dict[str, object], name: str, metric: str) -> str:
    text = _fmt(row[f"{name}_{metric}"], signed=True)
    low, high = row[f"{name}_{metric}_ci_low"], row[f"{name}_{metric}_ci_high"]
    if np.isfinite(low):
        text += f" [{_fmt(low, signed=True)}, {_fmt(high, signed=True)}]"
    return text


def comparison_markdown(
    rows: List[Dict[str, object]], comparison: ComparisonConfig, run_dir: str, confidence: float
) -> str:
    systems = [*comparison.references, RUN]
    contrasts = {f"{a}_vs_{b}": (LABELS[a], LABELS[b]) for a, b in CONTRASTS if a in systems and b in systems}
    lines: List[str] = []
    for system in dict.fromkeys(row["system"] for row in rows):
        mine = [row for row in rows if row["system"] == system]
        lines += [
            f"# `{system}` against the runs it controls for",
            "",
            "Each system's correlation with gold on the same pairs: the ones this run scored and every "
            f"other system answered. Differences carry a {comparison.bootstrap}-resample paired-bootstrap "
            f"{confidence:.0%} CI.",
            "",
            f"- **{LABELS[RUN]}**: {LEGEND[RUN]} (`{run_dir}`, `{system}`)",
        ]
        lines += [f"- **{LABELS[role]}**: {LEGEND[role]} (`{source.output_dir}`, `{source.model}`)"
                  for role, source in comparison.references.items()]
        for metric, title in (("spearman", "Spearman"), ("pearson", "Pearson")):
            lines += ["", f"## {title} with gold", ""]
            lines += _table(
                ["dataset", "pairs", *(LABELS[s] for s in systems), *(f"{a} − {b}" for a, b in contrasts.values())],
                [[str(row["dataset"]), str(row["n"]),
                  *(_fmt(row[f"{s}_{metric}"]) for s in systems),
                  *(_delta(row, name, metric) for name in contrasts)] for row in mine],
            )
        lines += [
            "## How far the scores moved",
            "",
            "Per pair of systems: the mean absolute difference between their scores on the judge's scale "
            f"(scale width in brackets) and the Spearman between their scores. **{LABELS[RUN]}** and "
            f"**{LABELS['matched']}** render the same prompt text, so only the images separate them.",
            "",
        ]
        lines += _table(
            ["dataset", *(f"{a} vs {b}: {stat}" for a, b in contrasts.values() for stat in ("mean abs Δ", "ρ"))],
            [[f"{row['dataset']} [{_fmt(row['scale_width'], 0)}]",
              *(cell for name in contrasts for cell in (_fmt(row[f"{name}_mean_abs_shift"], 3),
                                                         _fmt(row[f"{name}_agreement"])))]
             for row in mine],
        )
        if "image_following" in mine[0]:
            lines += [
                "## Do the scores follow the images shown?",
                "",
                "Donor = the pair whose images were shown. Gold vs donor gold checks the randomisation "
                f"(≈ 0); image following is the Spearman between (**{LABELS[RUN]}** − **{LABELS['matched']}**) "
                "and (donor gold − gold): positive when a score moves toward the similarity of the pair the "
                "images came from.",
                "",
            ]
            lines += _table(
                ["dataset", "gold vs donor gold", "score vs donor gold", "image following"],
                [[str(row["dataset"]), _fmt(row["gold_vs_donor_gold"], signed=True),
                  _fmt(row["run_vs_donor_gold"], signed=True), _fmt(row["image_following"], signed=True)]
                 for row in mine],
            )
        lines += [
            "## Reading it",
            "",
            f"A judge that ignores its images gives **{LABELS[RUN]}** = **{LABELS['matched']}** pair for "
            "pair: a difference whose CI covers 0, no score shift (ρ = 1) and no image following. A judge that "
            "reads them loses correlation when they belong to another pair, moves its scores, and moves "
            "them toward the donor's similarity. This tests whether the images are read, not whether "
            "they help: images that are read but redundant with the sentences add nothing to D2 and "
            "still cost here.",
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_comparison(config: ExperimentConfig) -> Dict[str, object]:
    """Compare every enabled judge on every dataset; write comparison.{csv,md,json}. Returns paths and rows."""
    comparison = config.comparison
    if comparison is None:
        raise ValueError(f"{config.source_path}: no `comparison:` section")
    if not config.enabled_judges:
        raise ValueError(f"{config.source_path}: no enabled judge to compare")
    if config.limit:
        _log.warning("--limit is ignored: the comparison uses exactly the pairs the run scored")
    runs = {judge.key: FusionSource(output_dir=config.output_dir, model=judge.key) for judge in config.enabled_judges}

    # Every prediction file first, so a missing run fails before anything is written.
    missing = [str(source.predictions(name)) for source in [*runs.values(), *comparison.references.values()]
               for name in config.datasets if not source.predictions(name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} prediction file(s) missing; compare reads finished runs, so check that this run "
            "finished and the output_dir/model of each run under `comparison:`:\n  " + "\n  ".join(missing)
        )

    manifest = Manifest(config=config, path=config.tagged("comparison", ".json"))
    manifest.payload["method"] = "comparison"
    rows: List[Dict[str, object]] = []
    sources: Dict[str, Dict[str, object]] = {}
    for name in config.datasets:
        dataset = load_dataset(name, config.data)
        manifest.record_dataset(dataset)
        sources[name] = {}
        references: Dict[str, pd.DataFrame] = {}
        for role, source in comparison.references.items():
            references[role], sources[name][role] = read_source(source, dataset)
        for key, source in runs.items():
            run, sources[name][key] = read_source(source, dataset)
            row = compare_dataset(key, dataset, run, references, comparison,
                                  config.metrics.bootstrap_seed, config.metrics.confidence)
            rows.append(row)
            _log.info("[%s | %s] %d pairs | ρ %s | other − own %+.4f%s", key, name, row["n"],
                      " ".join(f"{LABELS[r]} {row[f'{r}_spearman']:.4f}" for r in [*references, RUN]),
                      row[f"{RUN}_vs_matched_spearman"],
                      f" | image following {row['image_following']:+.4f}" if "image_following" in row else "")

    written = {"csv": config.tagged("comparison", ".csv"), "md": config.tagged("comparison", ".md")}
    pd.DataFrame(rows).to_csv(written["csv"], index=False)
    written["md"].write_text(
        comparison_markdown(rows, comparison, str(config.output_dir), config.metrics.confidence), encoding="utf-8"
    )
    manifest.payload["comparison"] = {
        "bootstrap": comparison.bootstrap,
        "references": {role: {"output_dir": str(s.output_dir), "model": s.model}
                       for role, s in comparison.references.items()},
        "sources": sources,
        "outputs": {kind: str(path) for kind, path in written.items()},
        "rows": rows,
    }
    manifest.finish()
    return {**written, "json": manifest.path, "rows": rows}
