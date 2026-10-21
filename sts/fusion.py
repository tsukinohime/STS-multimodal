"""Late fusion (condition LF): a text system and an image system, weighed per pair.

    fused = alpha * norm(text score) + (1 - alpha) * norm(image score)

Both inputs are prediction files that finished runs already wrote
(``<output_dir>/predictions/<model>/<dataset>.csv``), joined on ``pair_id``:
e.g. the jina text cosine (A0) with the jina image cosine (D1), or the text
judge (B1) with the image judge (D1). No model is loaded; a run is CPU only.

**Normalisation** (``fusion.normalization``), within each dataset and without
labels: ``zscore`` (default) puts both sources on mean 0 / sd 1, so alpha
weighs the two sources rather than their spreads (an image cosine varies far
less than a text cosine); ``rank`` uses fractional ranks; ``none`` keeps the
raw scores.

**Alpha** is read three ways in one run:

* *sweep*: every alpha of the grid on every dataset and dev set
  (``alpha_sweep.csv``), the curve from image only (0) to text only (1). An
  analysis; no number is picked from it.
* *fixed*: ``fixed_alpha`` (0.5), set before looking at any result.
* *dev*: for each evaluation dataset in ``select_alpha_on``, the alpha with the
  best mean Spearman over its dev sets, applied unchanged. Dev pairs never
  overlap the evaluation pairs, and no test split's gold scores enter a choice:
  CxC test from CxC val, SICK test from SICK train + trial, STS3k (no dev split
  of its own) from both.

**Pairs.** All systems of one fused pair are scored on the same pairs: those
both sources scored (a judge's degenerate answers have no score; they are
counted as ``n_degenerate``). The text-only and image-only endpoints are
written as systems of their own on those pairs, and the run checks that they
reproduce the sources' correlations. ``fusion_comparison`` adds a paired
bootstrap of each fused system's gain over the text-only one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .config import DevSet, ExperimentConfig, FusionConfig, FusionSource, FusionSystem, get_logger
from .data import STSDataset, load_dataset
from .hashing import sha256_file
from .manifest import Manifest
from .metrics import compute_correlations
from .pipeline import RunPaths, metric_rows, prediction_column_of, set_global_seed

_log = get_logger()

FUSED_COLUMN = "fused_score"
#: Per-pair columns copied from the text source into every fused prediction file.
_CARRIED = ("dataset", "split", "domain", "gold_score", "sentence1_id", "sentence2_id", "image1_id", "image2_id")
#: Roles of the systems one fused pair writes, in report order.
ROLES = ("text", "image", "fixed", "dev")


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #
def normalise(values: np.ndarray, method: str) -> np.ndarray:
    """One source's scores on one dataset, put on a common footing (labels unused)."""
    if method == "none":
        return values
    if method == "rank":
        return rankdata(values) / values.size
    if method == "zscore":
        spread = values.std()
        if not spread > 0:
            raise ValueError("a source gives every pair the same score; it cannot be z-scored")
        return (values - values.mean()) / spread
    raise ValueError(f"unknown normalisation {method!r}")


def fuse(text: np.ndarray, image: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * text + (1.0 - alpha) * image


@dataclass
class Joined:
    """One dataset (or dev set) as both sources scored it, indexed by pair_id."""

    name: str
    frame: pd.DataFrame       # _CARRIED + text_score, image_score, pair_truncated
    scored: np.ndarray        # both sources have a score for the pair

    @property
    def n_unscored(self) -> int:
        return int((~self.scored).sum())

    @property
    def gold(self) -> np.ndarray:
        return self.frame["gold_score"].to_numpy(np.float64)[self.scored]

    def raw(self, source: str) -> np.ndarray:
        return self.frame[f"{source}_score"].to_numpy(np.float64)[self.scored]

    def normalised(self, method: str) -> Tuple[np.ndarray, np.ndarray]:
        """Both sources on the scored pairs, each normalised within this set."""
        return normalise(self.raw("text"), method), normalise(self.raw("image"), method)

    def subset(self, name: str, keep: np.ndarray) -> "Joined":
        return Joined(name=name, frame=self.frame[keep], scored=self.scored[keep])


def _pair_keys(frame: pd.DataFrame) -> List[Tuple[str, str]]:
    """Pairs identified across datasets: sick-all and sick-test share sentence ids, not pair ids."""
    return list(zip(frame["sentence1_id"].astype(str), frame["sentence2_id"].astype(str)))


def _dataset_keys(dataset: STSDataset) -> Set[Tuple[str, str]]:
    return {(p.sentence1_id, p.sentence2_id) for p in dataset.pairs}


# --------------------------------------------------------------------------- #
# Reading the finished runs
# --------------------------------------------------------------------------- #
def read_source(source: FusionSource, dataset: STSDataset) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """One run's predictions on one dataset, checked against the current data file."""
    path = source.predictions(dataset.name)
    if not path.exists():
        raise FileNotFoundError(
            f"no predictions at {path}. Late fusion reads finished runs: check this source's "
            f"output_dir and model, and that its run covered {dataset.name}."
        )
    frame = pd.read_csv(path)
    column = prediction_column_of(frame)
    frame = frame.set_index("pair_id")
    if frame.index.has_duplicates:
        raise ValueError(f"{path}: duplicate pair ids")
    pairs = dataset.to_dataframe().set_index("pair_id")
    unknown = frame.index.difference(pairs.index)
    if len(unknown):
        raise ValueError(f"{path}: {len(unknown)} pair ids not in {dataset.name}, e.g. {unknown[0]}")
    gold = pairs.loc[frame.index, "gold_score"].to_numpy(np.float64)
    if not np.allclose(frame["gold_score"].to_numpy(np.float64), gold):
        raise ValueError(f"{path}: gold scores disagree with the current {dataset.name} data file")
    record = {"path": str(path), "sha256": sha256_file(path), "model": source.model,
              "column": column, "n_pairs": int(len(frame))}
    return frame.rename(columns={column: "score"}), record


def _truncated(frame: pd.DataFrame) -> pd.Series:
    if "pair_truncated" not in frame:
        return pd.Series(False, index=frame.index)
    return frame["pair_truncated"].fillna(False).astype(bool)


def join(system: FusionSystem, dataset: STSDataset) -> Tuple[Joined, Dict[str, Dict[str, object]]]:
    """The text and image runs' scores side by side; both must have scored the same pairs."""
    text, text_record = read_source(system.text, dataset)
    image, image_record = read_source(system.image, dataset)
    only_text = text.index.difference(image.index)
    only_image = image.index.difference(text.index)
    if len(only_text) or len(only_image):
        raise ValueError(
            f"{system.key} on {dataset.name}: the two runs scored different pairs ({len(text)} text, "
            f"{len(image)} image; {len(only_text)} only in text, {len(only_image)} only in image). "
            "Fuse runs over the same pairs: both full, or both with the same --limit."
        )
    image = image.loc[text.index]
    frame = pd.DataFrame(index=text.index)
    for column in _CARRIED:
        frame[column] = text[column] if column in text else ""
    frame["text_score"] = text["score"].astype(np.float64)
    frame["image_score"] = image["score"].astype(np.float64)
    frame["pair_truncated"] = _truncated(text) | _truncated(image)
    scored = frame["text_score"].notna().to_numpy() & frame["image_score"].notna().to_numpy()
    return Joined(dataset.name, frame, scored), {"text": text_record, "image": image_record}


def build_dev_sets(
    specs: Dict[str, DevSet], joined: Dict[str, Joined], datasets: Dict[str, STSDataset]
) -> Dict[str, Joined]:
    """Each dev set: its dataset as both sources scored it, minus every pair of its ``exclude`` datasets."""
    devs: Dict[str, Joined] = {}
    for name, spec in specs.items():
        if datasets[spec.dataset].split == "test":
            raise ValueError(f"dev set {name!r} is the test split {spec.dataset}: alpha is never chosen on a test split")
        excluded: Set[Tuple[str, str]] = set()
        for other in spec.exclude:
            excluded |= _dataset_keys(datasets[other])
        base = joined[spec.dataset]
        keep = np.array([key not in excluded for key in _pair_keys(base.frame)], dtype=bool)
        devs[name] = base.subset(name, keep)
        if devs[name].scored.sum() < 2:
            raise ValueError(f"dev set {name!r} has fewer than two scored pairs")
    return devs


# --------------------------------------------------------------------------- #
# Choosing alpha, and the paired comparison
# --------------------------------------------------------------------------- #
def select_alpha(
    devs: Sequence[Joined], alphas: Sequence[float], normalization: str
) -> Tuple[float, List[Dict[str, float]]]:
    """The alpha with the best mean Spearman over ``devs``; ties go to the larger alpha (more text)."""
    normalised = [dev.normalised(normalization) for dev in devs]
    table: List[Dict[str, float]] = []
    for alpha in alphas:
        rhos = [compute_correlations(fuse(t, i, alpha), dev.gold).spearman
                for dev, (t, i) in zip(devs, normalised)]
        table.append({"alpha": alpha, "dev_spearman": float(np.mean(rhos)),
                      **{dev.name: float(rho) for dev, rho in zip(devs, rhos)}})
    finite = [row for row in table if np.isfinite(row["dev_spearman"])]
    if not finite:
        raise ValueError(f"no alpha gives a finite Spearman on {[d.name for d in devs]}")
    best = max(finite, key=lambda row: (row["dev_spearman"], row["alpha"]))
    return float(best["alpha"]), table


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 0 else float("nan")


def paired_bootstrap(
    gold: np.ndarray,
    reference: np.ndarray,
    others: Dict[str, np.ndarray],
    resamples: int,
    seed: int,
    confidence: float,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Percentile CI of corr(other) - corr(reference), the same pairs resampled for both.

    Spearman is the Pearson of average ranks, re-ranked in every resample —
    what ``scipy.stats.spearmanr`` computes.
    """
    rng = np.random.default_rng(seed)
    n = gold.size
    deltas = {key: np.full((resamples, 2), np.nan) for key in others}
    for b in range(resamples):
        index = rng.integers(0, n, size=n)
        g = gold[index]
        g_rank = rankdata(g)
        ref = reference[index]
        ref_rho, ref_r = _pearson(rankdata(ref), g_rank), _pearson(ref, g)
        for key, values in others.items():
            v = values[index]
            deltas[key][b] = (_pearson(rankdata(v), g_rank) - ref_rho, _pearson(v, g) - ref_r)
    tail = 100.0 * (1.0 - confidence) / 2.0
    intervals: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for key, values in deltas.items():
        intervals[key] = {}
        for column, metric in enumerate(("spearman", "pearson")):
            finite = values[:, column][np.isfinite(values[:, column])]
            low, high = np.percentile(finite, [tail, 100.0 - tail]) if finite.size else (np.nan, np.nan)
            intervals[key][metric] = (float(low), float(high))
    return intervals


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #
def write_predictions(
    paths: RunPaths,
    key: str,
    system: FusionSystem,
    data: Joined,
    scores: np.ndarray,
    alpha: float,
    alpha_source: str,
    normalization: str,
) -> pd.DataFrame:
    """The standard prediction file; pairs a source left unscored keep a NaN score."""
    frame = data.frame
    fused = np.full(len(frame), np.nan)
    fused[data.scored] = scores
    out = pd.DataFrame({
        "pair_id": frame.index,
        "dataset": frame["dataset"].to_numpy(),
        "split": frame["split"].to_numpy(),
        "domain": frame["domain"].to_numpy(),
        "model": key,
        "gold_score": frame["gold_score"].to_numpy(),
        FUSED_COLUMN: fused,
        "alpha": alpha,
        "alpha_source": alpha_source,
        "normalization": normalization,
        "text_score": frame["text_score"].to_numpy(),
        "image_score": frame["image_score"].to_numpy(),
        "text_system": f"{system.text.output_dir.name}/{system.text.model}",
        "image_system": f"{system.image.output_dir.name}/{system.image.model}",
        "sentence1_id": frame["sentence1_id"].to_numpy(),
        "sentence2_id": frame["sentence2_id"].to_numpy(),
        "image1_id": frame["image1_id"].to_numpy(),
        "image2_id": frame["image2_id"].to_numpy(),
        "pair_truncated": frame["pair_truncated"].to_numpy(),
    })
    path = paths.prediction_file(key, data.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def _fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _table(header: List[str], rows: List[List[str]]) -> List[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|",
            *("| " + " | ".join(row) + " |" for row in rows)]


def _delta(row: Dict[str, object], role: str, metric: str) -> str:
    """Δ vs text with its CI, or a dash when the role was not computed for this dataset."""
    if f"{role}_delta_{metric}" not in row:
        return "—"
    text = _fmt(row[f"{role}_delta_{metric}"], signed=True)
    low, high = row[f"{role}_delta_{metric}_ci_low"], row[f"{role}_delta_{metric}_ci_high"]
    if np.isfinite(low):
        text += f" [{_fmt(low, signed=True)}, {_fmt(high, signed=True)}]"
    return text


def comparison_markdown(rows: List[Dict[str, object]], fusion: FusionConfig) -> str:
    lines = [
        "# Late fusion vs the text system",
        "",
        f"fused = alpha * text + (1 - alpha) * image, each source {fusion.normalization}-normalised "
        "within the dataset. `dev alpha` was chosen on the dev sets in brackets; Δ is the fused "
        f"system minus the text-only one on the same pairs, with a {fusion.bootstrap}-resample "
        "paired-bootstrap 95% CI.",
    ]
    for metric, name in (("spearman", "Spearman"), ("pearson", "Pearson")):
        for system in dict.fromkeys(row["system"] for row in rows):
            lines += ["", f"## {system} — {name}", ""]
            body = []
            for row in (r for r in rows if r["system"] == system):
                dev = (f"{_fmt(row['dev_alpha'], 2)} ({', '.join(row['dev_selected_on'])})"
                       if row.get("dev_alpha") is not None else "—")
                body.append([
                    str(row["dataset"]), str(row["n"]),
                    _fmt(row[f"text_{metric}"]), _fmt(row[f"image_{metric}"]),
                    _fmt(row[f"fixed_{metric}"]), _delta(row, "fixed", metric),
                    dev, _fmt(row.get(f"dev_{metric}", np.nan)), _delta(row, "dev", metric),
                ])
            lines += _table(
                ["dataset", "pairs", "text only", "image only", f"α={fusion.fixed_alpha:g}",
                 "Δ vs text", "dev α (chosen on)", "dev-α", "Δ vs text"],
                body,
            )
    return "\n".join(lines) + "\n"


def sweep_markdown(sweep: pd.DataFrame, fusion: FusionConfig, columns: int = 11) -> str:
    """Spearman over alpha per dataset; at most ``columns`` alphas shown (the CSV has the grid)."""
    alphas = sorted(sweep["alpha"].unique())
    step = max(1, (len(alphas) - 1) // (columns - 1))
    shown = alphas[::step] if alphas[::step][-1] == alphas[-1] else [*alphas[::step], alphas[-1]]
    lines = [
        "# Spearman over alpha (0 = image only, 1 = text only)",
        "",
        f"Each source {fusion.normalization}-normalised within the set. An analysis curve: no "
        "reported number is picked from it. Dev sets are the pairs alpha is chosen on.",
    ]
    for system, group in sweep.groupby("system", sort=False):
        lines += ["", f"## {system}", ""]
        body = []
        for (dataset, role), rows in group.groupby(["dataset", "role"], sort=False):
            by_alpha = rows.set_index("alpha")["spearman"]
            body.append([f"{dataset}{' (dev)' if role == 'dev' else ''}", str(int(rows["n"].iloc[0]))]
                        + [_fmt(by_alpha.get(a, np.nan)) for a in shown])
        lines += _table(["dataset", "pairs", *(f"{a:g}" for a in shown)], body)
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_fusion(config: ExperimentConfig) -> pd.DataFrame:
    """Fuse every enabled system on every dataset; write predictions, sweep, comparison. Returns metric rows."""
    fusion = config.fusion
    if fusion is None or not fusion.enabled_systems:
        raise ValueError(f"{config.source_path}: no enabled systems under `fusion:`")
    if config.limit:
        _log.warning("--limit is ignored: fusion uses exactly the pairs the source runs scored")
    set_global_seed(config.runtime.seed)

    targets = [t for t in fusion.select_alpha_on if t in config.datasets]
    for target in fusion.select_alpha_on:
        if target not in config.datasets:
            _log.warning("select_alpha_on: %s is not among this run's datasets; skipped", target)
    dev_specs = {name: fusion.dev_sets.get(name) or DevSet(dataset=name)
                 for target in targets for name in fusion.select_alpha_on[target]}
    joined_names = list(dict.fromkeys([*config.datasets, *(s.dataset for s in dev_specs.values())]))
    loaded_names = list(dict.fromkeys([*joined_names, *(d for s in dev_specs.values() for d in s.exclude)]))

    # Every source file first, so a missing run fails before anything is written.
    missing = [str(source.predictions(name)) for system in fusion.enabled_systems
               for source in (system.text, system.image) for name in joined_names
               if not source.predictions(name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} prediction file(s) missing; late fusion reads finished runs, so check the "
            "sources' output_dir and model in the config:\n  " + "\n  ".join(missing)
        )

    paths = RunPaths(config.output_dir)
    paths.ensure()
    manifest = Manifest(config=config, path=config.tagged("manifest", ".json"))
    manifest.payload["method"] = "late-fusion"
    datasets = {name: load_dataset(name, config.data) for name in loaded_names}
    for name in config.datasets:
        manifest.record_dataset(datasets[name])
        datasets[name].to_dataframe().to_csv(paths.pair_file(name), index=False)

    metrics: List[Dict[str, object]] = []
    sweep_rows: List[Dict[str, object]] = []
    comparison: List[Dict[str, object]] = []
    selections: Dict[str, Dict[str, object]] = {}
    sources: Dict[str, Dict[str, object]] = {}

    for system in fusion.enabled_systems:
        keys = fusion.output_keys(system)
        _log.info("[%s] text %s/%s + image %s/%s, %s-normalised", system.key,
                  system.text.output_dir, system.text.model, system.image.output_dir,
                  system.image.model, fusion.normalization)
        joined: Dict[str, Joined] = {}
        sources[system.key] = {}
        for name in joined_names:
            joined[name], sources[system.key][name] = join(system, datasets[name])
        devs = build_dev_sets(dev_specs, joined, datasets)

        # Alpha on dev pairs only, never on the pairs it is applied to.
        chosen: Dict[str, float] = {}
        selections[system.key] = {}
        for target in targets:
            names = fusion.select_alpha_on[target]
            target_keys = _dataset_keys(datasets[target])
            for name in names:
                overlap = sum(key in target_keys for key in _pair_keys(devs[name].frame))
                if overlap:
                    raise ValueError(f"dev set {name!r} shares {overlap} pairs with {target}; "
                                     f"add {target} to its exclude list")
            chosen[target], table = select_alpha([devs[n] for n in names], fusion.alphas, fusion.normalization)
            best = next(row for row in table if row["alpha"] == chosen[target])
            selections[system.key][target] = {
                "alpha": chosen[target], "selected_on": names, "dev_spearman": best["dev_spearman"],
                "n_dev_pairs": {n: int(devs[n].scored.sum()) for n in names},
            }
            _log.info("[%s] alpha for %s = %.2f (mean dev Spearman %.4f on %s)",
                      system.key, target, chosen[target], best["dev_spearman"], ", ".join(names))

        # The curve, on every evaluation dataset and every dev set that is not one of them.
        own = [(n, d, "dev") for n, d in devs.items() if dev_specs[n].exclude or n not in config.datasets]
        for name, data, role in [*((n, joined[n], "evaluation") for n in config.datasets), *own]:
            text, image = data.normalised(fusion.normalization)
            for alpha in fusion.alphas:
                correlations = compute_correlations(fuse(text, image, alpha), data.gold)
                sweep_rows.append({"system": system.key, "dataset": name, "role": role, "alpha": alpha,
                                   "n": correlations.n, "spearman": correlations.spearman,
                                   "pearson": correlations.pearson})

        for name in config.datasets:
            started = time.time()
            data = joined[name]
            text, image = data.normalised(fusion.normalization)
            scores = {"text": text, "image": image, "fixed": fuse(text, image, fusion.fixed_alpha)}
            alphas = {"text": 1.0, "image": 0.0, "fixed": fusion.fixed_alpha}
            origin = {"text": "text only", "image": "image only", "fixed": "fixed in advance"}
            if name in chosen:
                scores["dev"] = fuse(text, image, chosen[name])
                alphas["dev"] = chosen[name]
                origin["dev"] = "chosen on " + ", ".join(fusion.select_alpha_on[name])

            # The endpoints must reproduce the sources: normalising may not reorder anything.
            gold = data.gold
            for role in ("text", "image"):
                expected = compute_correlations(data.raw(role), gold).spearman
                got = compute_correlations(scores[role], gold).spearman
                if not abs(expected - got) < 1e-9:
                    raise RuntimeError(f"{keys[role]} on {name}: Spearman {got} != source's {expected}")

            correlations = {}
            for role, values in scores.items():
                frame = write_predictions(paths, keys[role], system, data, values, alphas[role],
                                          origin[role], fusion.normalization)
                rows = metric_rows(frame[frame[FUSED_COLUMN].notna()], name, keys[role], config,
                                   prediction_column=FUSED_COLUMN)
                for row in rows:
                    row["n_degenerate"] = data.n_unscored
                metrics.extend(rows)
                correlations[role] = compute_correlations(values, gold)

            others = {role: scores[role] for role in ("fixed", "dev") if role in scores}
            intervals = (paired_bootstrap(gold, scores["text"], others, fusion.bootstrap,
                                          config.metrics.bootstrap_seed, config.metrics.confidence)
                         if fusion.bootstrap else {})
            row: Dict[str, object] = {"system": system.key, "dataset": name, "n": int(gold.size),
                                      "n_unscored": data.n_unscored,
                                      "fixed_alpha": fusion.fixed_alpha,
                                      "dev_alpha": chosen.get(name),
                                      "dev_selected_on": fusion.select_alpha_on.get(name, []) if name in chosen else []}
            for metric in ("spearman", "pearson"):
                for role, value in correlations.items():
                    row[f"{role}_{metric}"] = getattr(value, metric)
                for role in others:
                    row[f"{role}_delta_{metric}"] = getattr(correlations[role], metric) - getattr(correlations["text"], metric)
                    low, high = intervals.get(role, {}).get(metric, (np.nan, np.nan))
                    row[f"{role}_delta_{metric}_ci_low"], row[f"{role}_delta_{metric}_ci_high"] = low, high
            comparison.append(row)
            manifest.record_run({
                "model": system.key, "dataset": name, "n_pairs": int(len(data.frame)),
                "n_unique_texts": len(datasets[name].unique_texts()),
                "runtime_seconds": round(time.time() - started, 2),
                "n_truncated_pairs": int(data.frame["pair_truncated"].sum()),
                "n_unscored": data.n_unscored, "dev_alpha": chosen.get(name),
            })
            _log.info("[%s | %s] %d pairs | ρ text %.4f image %.4f α=%g %.4f%s", system.key, name, gold.size,
                      correlations["text"].spearman, correlations["image"].spearman, fusion.fixed_alpha,
                      correlations["fixed"].spearman,
                      f" dev α={chosen[name]:g} {correlations['dev'].spearman:.4f}" if name in chosen else "")

        for role, key in keys.items():
            if role == "dev" and not chosen:
                continue
            manifest.payload["models"][key] = {
                "model_id": f"late fusion: α·{system.text.model} + (1−α)·{system.image.model}",
                "revision": None,
                "role": role,
                "normalization": fusion.normalization,
                "text_source": str(system.text.output_dir / "predictions" / system.text.model),
                "image_source": str(system.image.output_dir / "predictions" / system.image.model),
            }

    sweep = pd.DataFrame(sweep_rows)
    written = {
        "sweep_csv": config.tagged("alpha_sweep", ".csv"),
        "sweep_md": config.tagged("alpha_sweep", ".md"),
        "comparison_csv": config.tagged("fusion_comparison", ".csv"),
        "comparison_md": config.tagged("fusion_comparison", ".md"),
    }
    sweep.to_csv(written["sweep_csv"], index=False)
    written["sweep_md"].write_text(sweep_markdown(sweep, fusion), encoding="utf-8")
    pd.DataFrame(comparison).assign(
        dev_selected_on=lambda f: f["dev_selected_on"].map(lambda names: "+".join(names))
    ).to_csv(written["comparison_csv"], index=False)
    written["comparison_md"].write_text(comparison_markdown(comparison, fusion), encoding="utf-8")

    manifest.payload["fusion"] = {
        "normalization": fusion.normalization,
        "alphas": fusion.alphas,
        "fixed_alpha": fusion.fixed_alpha,
        "bootstrap": fusion.bootstrap,
        "dev_sets": {name: {"dataset": spec.dataset, "exclude": spec.exclude} for name, spec in dev_specs.items()},
        "selections": selections,
        "sources": sources,
        "outputs": {name: str(path) for name, path in written.items()},
    }
    manifest.finish()
    for path in written.values():
        _log.info("-> %s", path)
    return pd.DataFrame(metrics)
