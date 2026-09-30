"""Post-hoc breakdown by pair category: SICK's NLI label, or any other dataset's domain.

Reads prediction CSVs already on disk — judge or embedding, any mix — and
gives each pair a label: for SICK, its entailment annotation, joined by
``pair_id``; for everything else, the pair table's ``domain`` (CxC's
``sampling_method``: ``c2c_cocaption`` / ``c2c_isim``). Nothing is re-scored:
the per-pair predictions are the experiment, and grouping them is bookkeeping,
so the numbers are exactly those of the original run.

Per system and label it reports:

* ``spearman`` / ``pearson`` **within** the label. The labels cut SICK's gold
  range sharply (the middle half of ENTAILMENT spans 4.4–4.8), so these are low
  for every system; compare them across systems, not with the pooled value.
* ``spearman_without`` — the pooled Spearman with this label's pairs removed.
  The label whose removal closes a system's gap to another is the one where
  that system loses correlation.
* ``rank_offset`` — mean of rank(pred) − rank(gold) over the label's pairs,
  both ranks as a fraction of the pooled set. Scale-free and zero over all
  pairs; negative means the system places this label lower than the annotators
  did. Within-label correlation cannot see this error, the pooled one can.
* ``rank_error_share`` — the label's share of the system's squared rank error
  Σ(rank(pred) − rank(gold))², the quantity Spearman's ρ is one minus (up to
  the tie correction). Set it against the label's share of pairs.
* ``pred_mean`` next to ``gold_mean`` in native units — directly comparable for
  a judge answering on the dataset's own scale (SICK 1–5, CxC 0–5), not for
  cosine.

Per system it also reports ``spearman_offset_removed``: the pooled Spearman
after subtracting each label's mean rank offset. It uses the gold labels, so it
is a diagnostic, not a result. Its gap to the plain Spearman is the part of the
loss that comes from placing whole labels at the wrong height rather than from
ranking pairs inside them.

All systems are scored on the same pairs: those every system has a finite
prediction for.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import ExperimentConfig, get_logger
from .data import load_dataset
from .data.sick import load_sick_labels
from .metrics import compute_correlations
from .pipeline import prediction_column_of

_log = get_logger()

ALL_LABEL = "ALL"


@dataclass(frozen=True)
class System:
    """One set of predictions for one dataset."""

    key: str
    path: Path
    column: str


# --------------------------------------------------------------------------- #
# Finding and aligning predictions
# --------------------------------------------------------------------------- #
def find_systems(
    config: ExperimentConfig,
    dataset_name: str,
    baselines: Sequence[Path] = (),
    baseline_models: Optional[Sequence[str]] = None,
) -> List[System]:
    """The config's enabled judges/models, then every model under each baseline dir."""
    found: Dict[str, System] = {}

    def add(key: str, path: Path) -> None:
        if key in found:
            raise ValueError(f"system {key!r} found twice: {found[key].path} and {path}")
        column = prediction_column_of(pd.read_csv(path, nrows=1))
        found[key] = System(key=key, path=path, column=column)

    for key in [j.key for j in config.enabled_judges] + [m.key for m in config.enabled_models]:
        path = config.output_dir / "predictions" / key / f"{dataset_name}.csv"
        if path.exists():
            add(key, path)
        else:
            _log.warning("[%s] no predictions at %s", key, path)

    wanted = set(baseline_models) if baseline_models else None
    for root in baselines:
        paths = sorted(Path(root).glob(f"predictions/*/{dataset_name}.csv"))
        if not paths:
            _log.warning("no %s predictions under %s", dataset_name, Path(root) / "predictions")
        for path in paths:
            key = path.parent.name
            if wanted is None or key in wanted:
                add(key, path)
    return list(found.values())


def aligned_frame(
    config: ExperimentConfig, dataset_name: str, systems: Sequence[System], label_column: str
) -> Optional[Dict[str, object]]:
    """One row per pair every system scored: label, gold, sentences, one column per system.

    ``None`` when the dataset has fewer than two categories (STS3k has none).
    """
    dataset = load_dataset(dataset_name, config.data)
    pairs = dataset.to_dataframe().set_index("pair_id")
    if dataset.name.startswith("sick-"):
        labels = load_sick_labels(config.data.sick_path, dataset.split, label_column)
    else:
        label_column = str(dataset.notes.get("domain_column") or "domain")
        labels = pairs["domain"].replace("", np.nan)
    if labels.nunique() < 2:
        _log.warning("%s: fewer than two %s values, nothing to break down", dataset_name, label_column)
        return None

    frame = pairs[["gold_score", "sentence1", "sentence2"]].join(labels.rename("label"), how="left")
    if frame["label"].isna().any():
        raise ValueError(f"{int(frame['label'].isna().sum())} {dataset_name} pairs have no {label_column}")

    dropped: Dict[str, int] = {}
    for system in systems:
        predictions = pd.read_csv(system.path, usecols=["pair_id", "gold_score", system.column])
        predictions = predictions.set_index("pair_id")
        unknown = predictions.index.difference(frame.index)
        if len(unknown):
            raise ValueError(f"{system.path}: {len(unknown)} pair ids not in {dataset_name}, e.g. {unknown[0]}")
        gold = frame.loc[predictions.index, "gold_score"].to_numpy()
        if not np.allclose(predictions["gold_score"].to_numpy(), gold):
            raise ValueError(f"{system.path}: gold scores disagree with the current {dataset_name} file")
        frame[system.key] = predictions[system.column].reindex(frame.index)
        dropped[system.key] = int(frame[system.key].isna().sum())

    keep = frame[[s.key for s in systems]].notna().all(axis=1)
    if not keep.all():
        _log.warning("%s: scoring %d of %d pairs, the ones every system predicted",
                     dataset_name, int(keep.sum()), len(frame))
    return {"frame": frame[keep].copy(), "n_dataset": len(frame), "missing": dropped,
            "label_column": label_column}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def _fraction_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values) / len(values)


def breakdown_rows(frame: pd.DataFrame, systems: Sequence[System]) -> pd.DataFrame:
    """ALL + one row per label, for every system."""
    gold = frame["gold_score"].to_numpy(dtype=np.float64)
    labels = frame["label"].to_numpy()
    gold_rank = _fraction_ranks(gold)
    rows: List[Dict[str, object]] = []

    for system in systems:
        pred = frame[system.key].to_numpy(dtype=np.float64)
        offset = _fraction_ranks(pred) - gold_rank
        squared_error = float(np.sum(offset ** 2))
        overall = compute_correlations(pred, gold)

        label_offset = {label: float(offset[labels == label].mean()) for label in np.unique(labels)}
        adjusted = _fraction_ranks(pred) - np.array([label_offset[label] for label in labels])

        rows.append({
            "system": system.key,
            "label": ALL_LABEL,
            "n": overall.n,
            "spearman": overall.spearman,
            "pearson": overall.pearson,
            "spearman_without": float("nan"),
            "gold_mean": float(gold.mean()),
            "gold_std": float(gold.std()),
            "pred_mean": float(pred.mean()),
            "rank_offset": float(offset.mean()),
            "rank_error_share": 1.0,
            "spearman_offset_removed": compute_correlations(adjusted, gold).spearman,
            "prediction_column": system.column,
        })
        for label in sorted(label_offset):
            inside = labels == label
            within = compute_correlations(pred[inside], gold[inside])
            rows.append({
                "system": system.key,
                "label": label,
                "n": within.n,
                "spearman": within.spearman,
                "pearson": within.pearson,
                "spearman_without": compute_correlations(pred[~inside], gold[~inside]).spearman,
                "gold_mean": float(gold[inside].mean()),
                "gold_std": float(gold[inside].std()),
                "pred_mean": float(pred[inside].mean()),
                "rank_offset": label_offset[label],
                "rank_error_share": float(np.sum(offset[inside] ** 2)) / squared_error,
                "spearman_offset_removed": float("nan"),
                "prediction_column": system.column,
            })
    return pd.DataFrame(rows)


def worst_pairs(frame: pd.DataFrame, systems: Sequence[System], per_label: int) -> pd.DataFrame:
    """Per system and label, the pairs whose rank is furthest from gold, with every system's view."""
    gold_rank = _fraction_ranks(frame["gold_score"].to_numpy(dtype=np.float64))
    offsets = {
        s.key: _fraction_ranks(frame[s.key].to_numpy(dtype=np.float64)) - gold_rank for s in systems
    }
    base = frame.reset_index().rename(columns={"index": "pair_id"})
    for s in systems:
        base[f"rank_offset__{s.key}"] = offsets[s.key]
    base = base.rename(columns={s.key: f"pred__{s.key}" for s in systems})

    parts = []
    for system in systems:
        column = f"rank_offset__{system.key}"
        for _, group in base.groupby("label", sort=True):
            picked = group.reindex(group[column].abs().sort_values(ascending=False).index).head(per_label)
            parts.append(picked.assign(worst_for=system.key))
    if not parts:
        return pd.DataFrame()
    leading = ["worst_for", "label", "pair_id", "gold_score", "sentence1", "sentence2"]
    result = pd.concat(parts, ignore_index=True)
    return result[leading + [c for c in result.columns if c not in leading]]


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines + [""]


def build_markdown(
    dataset_name: str,
    label_column: str,
    systems: Sequence[System],
    rows: pd.DataFrame,
    n_scored: int,
    n_dataset: int,
    missing: Dict[str, int],
) -> str:
    keys = [s.key for s in systems]
    labels = [label for label in rows["label"].drop_duplicates() if label != ALL_LABEL]
    cell = rows.set_index(["system", "label"])

    def row_of(label: str, metric: str, **fmt) -> List[str]:
        return [_fmt(cell.loc[(key, label), metric], **fmt) for key in keys]

    first = cell.xs(keys[0], level="system")
    lines = [
        f"# {dataset_name} by `{label_column}`",
        "",
        f"Pairs scored: **{n_scored:,}** of {n_dataset:,}, the ones every system predicted "
        f"(missing per system: {', '.join(f'{k} {v}' for k, v in missing.items())}).",
        "",
        "Systems:",
        "",
    ]
    lines += [f"- `{s.key}` — `{s.column}` from `{s.path}`" for s in systems]
    lines += ["", "## Spearman within each label", ""]
    lines += _table(
        ["label", "n", "gold mean ± sd"] + keys,
        [
            [label, str(int(first.loc[label, "n"])),
             f"{first.loc[label, 'gold_mean']:.2f} ± {first.loc[label, 'gold_std']:.2f}"]
            + row_of(label, "spearman")
            for label in [ALL_LABEL] + labels
        ],
    )
    lines += [
        "A label spanning a narrow slice of the gold scale (small sd) gives low within-label "
        "values for every system. Compare systems within a row.",
        "",
        "## Pooled Spearman with one label removed",
        "",
    ]
    lines += _table(
        ["removed", "pairs left"] + keys,
        [["none", str(n_scored)] + row_of(ALL_LABEL, "spearman")]
        + [[label, str(n_scored - int(first.loc[label, "n"]))] + row_of(label, "spearman_without")
           for label in labels],
    )
    lines += [
        "## Mean rank offset: rank(pred) − rank(gold), as a fraction of all pairs",
        "",
        "Negative: the system ranks this label's pairs lower than the annotators did.",
        "",
    ]
    lines += _table(["label"] + keys, [[label] + row_of(label, "rank_offset", signed=True) for label in labels])
    lines += ["## Share of squared rank error", ""]
    lines += _table(
        ["label", "share of pairs"] + keys,
        [[label, _fmt(first.loc[label, "n"] / n_scored, 3)] + row_of(label, "rank_error_share", digits=3)
         for label in labels],
    )
    lines += [
        "## Pooled Spearman with each label's mean rank offset removed",
        "",
        "Diagnostic only (it uses the gold labels). The gain over the plain value is the "
        "correlation lost to whole labels sitting at the wrong height.",
        "",
    ]
    lines += _table(
        ["system", "spearman", "offset removed", "gain"],
        [
            [key, _fmt(cell.loc[(key, ALL_LABEL), "spearman"]),
             _fmt(cell.loc[(key, ALL_LABEL), "spearman_offset_removed"]),
             _fmt(cell.loc[(key, ALL_LABEL), "spearman_offset_removed"]
                  - cell.loc[(key, ALL_LABEL), "spearman"], signed=True)]
            for key in keys
        ],
    )
    judges = [s.key for s in systems if s.column == "expected_score"]
    if judges:
        lines += ["## Judge score vs gold, on the dataset's own scale", ""]
        lines += _table(
            ["label", "gold mean"] + judges,
            [[label, _fmt(first.loc[label, "gold_mean"], 2)]
             + [_fmt(cell.loc[(key, label), "pred_mean"], 2) for key in judges]
             for label in [ALL_LABEL] + labels],
        )
    return "\n".join(lines)


def run_breakdown(
    config: ExperimentConfig,
    datasets: Sequence[str],
    baselines: Sequence[Path] = (),
    baseline_models: Optional[Sequence[str]] = None,
    label_column: str = "entailment_label",
    per_label: int = 10,
) -> List[Path]:
    """Write ``breakdown/<dataset>__<label>.{csv,md}`` and ``__worst.csv``; return the paths."""
    directory = config.output_dir / "breakdown"
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name in datasets:
        systems = find_systems(config, name, baselines, baseline_models)
        if not systems:
            _log.error("%s: no predictions found, skipped", name)
            continue
        aligned = aligned_frame(config, name, systems, label_column)
        if aligned is None:
            continue
        frame = aligned["frame"]
        rows = breakdown_rows(frame, systems)
        markdown = build_markdown(
            name, aligned["label_column"], systems, rows, len(frame), aligned["n_dataset"], aligned["missing"]
        )

        stem = directory / f"{name}__{aligned['label_column']}"
        paths = [stem.with_suffix(".csv"), stem.with_suffix(".md"), stem.with_name(stem.name + "__worst.csv")]
        rows.to_csv(paths[0], index=False)
        paths[1].write_text(markdown, encoding="utf-8")
        worst_pairs(frame, systems, per_label).to_csv(paths[2], index=False)
        print("\n" + markdown)
        _log.info("%s breakdown -> %s", name, ", ".join(str(p) for p in paths))
        written.extend(paths)
    return written
