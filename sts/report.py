"""Result aggregation: metrics CSV/JSON plus the human-readable summary.

The macro average is unweighted across the datasets listed in
``metrics.macro_average_over`` — each corpus counts once regardless of size, and
overlapping splits must not both be listed (``sick-all`` contains ``sick-test``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ExperimentConfig, get_logger
from .metrics import macro_average

_log = get_logger()

MACRO_ROW = "MACRO-AVG"


def add_macro_average(metrics: pd.DataFrame, over: List[str]) -> pd.DataFrame:
    """Append one unweighted macro-average row per model."""
    if metrics.empty:
        return metrics

    overall = metrics[metrics["domain"] == "ALL"]
    present = [d for d in over if d in set(overall["dataset"])]
    missing = [d for d in over if d not in set(overall["dataset"])]
    if missing:
        _log.warning("Macro average skips datasets not in this run: %s", missing)
    if not present:
        return metrics

    rows = []
    for model, group in overall[overall["dataset"].isin(present)].groupby("model", sort=False):
        rows.append({
            "model": model,
            "dataset": MACRO_ROW,
            "split": "",
            "domain": "ALL",
            "n": int(group["n"].sum()),
            "spearman": macro_average(group["spearman"]),
            "pearson": macro_average(group["pearson"]),
            "n_truncated_pairs": int(group["n_truncated_pairs"].sum()),
            "macro_over": "|".join(present),
        })
    return pd.concat([metrics, pd.DataFrame(rows)], ignore_index=True)


def write_metrics(metrics: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.tagged("metrics", ".csv")
    json_path = config.tagged("metrics", ".json")
    metrics.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            json.loads(metrics.to_json(orient="records")), indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    _log.info("Metrics -> %s and %s", csv_path, json_path)
    return {"csv": csv_path, "json": json_path}


def _format(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def _markdown_table(header: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    lines = [
        "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
        "|" + "|".join("-" * (w + 2) for w in widths) + "|",
    ]
    lines += ["| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _ci_suffix(row) -> str:
    low, high = row.get("spearman_ci_low"), row.get("spearman_ci_high")
    if low is None or (isinstance(low, float) and np.isnan(low)):
        return ""
    return f" [{low:.3f}, {high:.3f}]"


def build_markdown_summary(
    metrics: pd.DataFrame,
    config: ExperimentConfig,
    manifest: Optional[dict] = None,
) -> str:
    manifest = manifest or {}
    resolved = manifest.get("resolved", {})
    hardware = manifest.get("hardware", {})
    versions = manifest.get("versions", {})
    runs = manifest.get("runs", [])
    models_meta = manifest.get("models", {})
    datasets_meta = manifest.get("datasets", {})

    lines: List[str] = [
        f"# {config.name} — text-only zero-shot STS",
        "",
        "Frozen pretrained encoders, L2-normalised embeddings, cosine similarity as the "
        "prediction. No training, fine-tuning, regression, calibration or test-set tuning.",
        "",
        "## Run",
        "",
    ]
    lines += [
        f"- **Config**: `{config.source_path}` (sha256 `{manifest.get('config_sha256', '?')[:16]}`)",
        f"- **Command**: `{manifest.get('command', '?')}`",
        f"- **Started / finished**: {manifest.get('run_started', '?')} → {manifest.get('run_finished', '?')}",
        f"- **Total runtime**: {manifest.get('total_runtime_seconds', '?')} s",
        f"- **Device / dtype**: {resolved.get('device', '?')} / {resolved.get('dtype', '?')}",
        f"- **Batch size / max_length**: {resolved.get('batch_size', '?')} / {resolved.get('max_length', '?')}",
        f"- **Seed**: {manifest.get('seed', '?')}",
        f"- **Hardware**: {hardware.get('gpu_name', hardware.get('machine', '?'))} on {hardware.get('platform', '?')}",
        f"- **torch / transformers**: {versions.get('torch', '?')} / {versions.get('transformers', '?')}",
        f"- **Git**: {(manifest.get('git') or {}).get('commit', '?')}"
        f"{' (dirty)' if (manifest.get('git') or {}).get('dirty') else ''}",
        "",
    ]

    # ---- headline table ----
    lines += ["## Main results — Spearman's rho (Pearson's r)", ""]
    overall = metrics[metrics["domain"] == "ALL"]
    dataset_order = [d for d in config.datasets if d in set(overall["dataset"])]
    if MACRO_ROW in set(overall["dataset"]):
        dataset_order.append(MACRO_ROW)

    header = ["model"] + dataset_order
    rows = []
    for model in overall["model"].drop_duplicates():
        row = [str(model)]
        for dataset in dataset_order:
            match = overall[(overall["model"] == model) & (overall["dataset"] == dataset)]
            if match.empty:
                row.append("—")
            else:
                record = match.iloc[0]
                row.append(
                    f"{_format(record['spearman'])} ({_format(record['pearson'])})"
                    + _ci_suffix(record)
                )
        rows.append(row)
    lines.append(_markdown_table(header, rows))
    lines += [
        "",
        f"`{MACRO_ROW}` is the **unweighted** mean of the per-dataset Spearman/Pearson values "
        f"over `{', '.join(config.metrics.macro_average_over)}`. Datasets are never pooled into "
        "one global correlation.",
        "",
    ]

    # ---- per-dataset detail ----
    lines += ["## Per dataset", ""]
    detail_header = ["model", "dataset", "split", "domain", "n", "spearman", "pearson", "truncated pairs"]
    detail_rows = []
    for _, record in metrics.sort_values(["dataset", "model", "domain"]).iterrows():
        if record["dataset"] == MACRO_ROW:
            continue
        detail_rows.append([
            str(record["model"]), str(record["dataset"]), str(record["split"]),
            str(record["domain"]), str(int(record["n"])),
            _format(record["spearman"]) + _ci_suffix(record),
            _format(record["pearson"]),
            str(int(record.get("n_truncated_pairs", 0) or 0)),
        ])
    lines.append(_markdown_table(detail_header, detail_rows))

    # ---- category results ----
    category_rows = metrics[(metrics["domain"] != "ALL") & (metrics["dataset"] != MACRO_ROW)]
    lines += ["", "## Category / domain breakdown", ""]
    if category_rows.empty:
        lines += [
            "No dataset in this run ships a per-pair category column, so only overall "
            "results are available.",
            "",
        ]
    else:
        by_dataset = sorted(category_rows["dataset"].unique())
        lines.append(
            "Categories come from the source files: CxC `sampling_method` "
            "(`c2c_cocaption` = both captions describe the same COCO image, `c2c_isim` = "
            "different but visually similar images), SICK `sentence_A/B_dataset`. "
            f"Datasets with categories here: {', '.join(by_dataset)}."
        )
        lines.append("")

    # ---- runtime / truncation ----
    lines += ["## Runtime and cache", ""]
    if runs:
        runtime_rows = [
            [
                str(r["model"]), str(r["dataset"]), str(r["n_pairs"]), str(r["n_unique_texts"]),
                f"{r['runtime_seconds']:.1f}", str(r.get("pairs_per_second", "—")),
                f"{(r.get('cache') or {}).get('hit_rate', 0):.2f}",
                str(r.get("n_truncated_pairs", 0)),
            ]
            for r in runs
        ]
        lines.append(_markdown_table(
            ["model", "dataset", "pairs", "unique texts", "seconds", "pairs/s",
             "cache hit rate", "truncated pairs"],
            runtime_rows,
        ))
    else:
        lines.append("_All (model, dataset) combinations were resumed from existing predictions._\n")

    total_truncated = int(metrics[metrics["domain"] == "ALL"]["n_truncated_pairs"].sum())
    lines += [
        "",
        f"**Truncation**: {total_truncated} pair(s) had at least one sentence longer than "
        f"`max_length={resolved.get('max_length', '?')}` tokens and were truncated by the "
        "tokenizer. Per-pair flags are in the prediction CSVs "
        "(`sentence1_truncated`, `sentence2_truncated`, `pair_truncated`).",
        "",
    ]

    # ---- provenance ----
    lines += ["## Models", ""]
    if models_meta:
        lines.append(_markdown_table(
            ["key", "model id", "revision", "task", "prompt prefix", "pooling", "dtype"],
            [
                [
                    key,
                    str(meta.get("model_id", "")),
                    str(meta.get("revision") or "—")[:12],
                    str(meta.get("task") or "—"),
                    repr(meta.get("prompt_prefix", "")) if meta.get("prompt_prefix") else "—",
                    str(meta.get("pooling", "—")),
                    str(meta.get("dtype", "—")),
                ]
                for key, meta in models_meta.items()
            ],
        ))
    else:
        lines.append("_No model was loaded (fully resumed run)._\n")

    lines += ["", "## Data", ""]
    if datasets_meta:
        lines.append(_markdown_table(
            ["dataset", "split", "pairs", "rows in source", "gold range", "source sha256"],
            [
                [
                    name,
                    str(meta.get("split", "")),
                    str(meta.get("n_pairs", "")),
                    str(meta.get("n_rows_in_source", "")),
                    str(meta.get("score_range", "")),
                    ", ".join(f"{Path(p).name}:{h[:8]}" for p, h in (meta.get("source_files") or {}).items()),
                ]
                for name, meta in datasets_meta.items()
            ],
        ))

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python -m sts.cli run --config {config.source_path}",
        "```",
        "",
    ]
    return "\n".join(lines)


def write_summary(
    metrics: pd.DataFrame,
    config: ExperimentConfig,
    manifest: Optional[dict] = None,
) -> Path:
    path = config.tagged("summary", ".md")
    path.write_text(build_markdown_summary(metrics, config, manifest), encoding="utf-8")
    _log.info("Summary -> %s", path)
    return path


def print_console_table(metrics: pd.DataFrame) -> None:
    overall = metrics[metrics["domain"] == "ALL"]
    if overall.empty:
        print("(no results)")
        return
    header = f"{'model':<24}{'dataset':<14}{'n':>8}{'spearman':>11}{'pearson':>10}"
    print("\n" + header)
    print("-" * len(header))
    for _, row in overall.iterrows():
        print(
            f"{str(row['model']):<24}{str(row['dataset']):<14}{int(row['n']):>8}"
            f"{row['spearman']:>11.4f}{row['pearson']:>10.4f}"
        )
    print()
