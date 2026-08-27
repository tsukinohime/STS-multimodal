"""Typed configuration loaded from a single YAML file.

One YAML controls a whole experiment: which corpora, which frozen encoders,
inference settings, metric settings and where the artefacts land. Everything
that can change a number is captured here so it can be hashed into the manifest.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

_LOG_FORMAT = "[%(levelname)s] %(asctime)s %(message)s"


def get_logger(name: str = "sts") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("STS_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


_log = get_logger()


def resolve_device(device: str = "auto") -> str:
    """Resolve ``auto`` to the best available torch device (cuda > mps > cpu)."""
    if device and device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> str:
    """Resolve ``auto`` to a concrete dtype name.

    bfloat16 on CUDA (what the jina-v5 checkpoints ship as, and what H100s run
    natively), float32 elsewhere. MPS does have bfloat16 kernels but coverage is
    patchier than on CUDA and the models here are small enough that fp32 costs
    little on a laptop — set ``dtype: bfloat16`` explicitly to override.
    """
    if dtype and dtype != "auto":
        return dtype
    return "bfloat16" if device == "cuda" else "float32"


def torch_dtype(name: str):
    import torch

    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"unsupported dtype {name!r}; use one of {sorted(mapping)}")
    return mapping[name]


def _expand(path: Any, base: Path) -> Path:
    """Resolve a config path, honouring ``~`` and ``$VARS``, relative to the repo."""
    text = os.path.expandvars(os.path.expanduser(str(path)))
    candidate = Path(text)
    return candidate if candidate.is_absolute() else (base / candidate)


@dataclass
class DataPaths:
    root: Path = REPO_ROOT / "data" / "raw"
    sick_path: Optional[Path] = None
    cxc_dir: Optional[Path] = None
    coco_karpathy_json: Optional[Path] = None
    sts3k_path: Optional[Path] = None

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.sick_path = Path(self.sick_path) if self.sick_path else self.root / "sick" / "SICK.txt"
        self.cxc_dir = Path(self.cxc_dir) if self.cxc_dir else self.root / "cxc" / "data"
        self.coco_karpathy_json = (
            Path(self.coco_karpathy_json)
            if self.coco_karpathy_json
            else self.root / "coco_karpathy" / "dataset_coco.json"
        )
        self.sts3k_path = (
            Path(self.sts3k_path) if self.sts3k_path else self.root / "sts3k" / "STS3k_all.txt"
        )


@dataclass
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "auto"
    batch_size: int = 32
    max_length: int = 512
    cache_dir: Path = REPO_ROOT / ".cache" / "embeddings"
    use_cache: bool = True
    #: Unique texts encoded between cache flushes — the resume granularity.
    flush_every: int = 2048
    progress: bool = True
    seed: int = 42

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)


@dataclass
class ModelSpec:
    """One frozen encoder. ``key`` names it in every output table."""

    key: str
    model_id: str
    backend: str = "jina-v5"          # jina-v5 | sentence-transformers | hf
    task: Optional[str] = None        # jina-v5 task adapter, e.g. "text-matching"
    prompt_name: Optional[str] = None # jina-v5 prompt: "document" | "query"
    prompt_prefix: str = ""           # literal prefix for the ST / HF backends
    modality: Optional[str] = None    # jina-v5-omni: "text" loads the text tower only
    truncate_dim: Optional[int] = None  # None = full embedding dimension
    pooling: str = "mean"             # hf backend only: mean | cls
    batch_size: Optional[int] = None  # overrides RuntimeConfig.batch_size
    max_length: Optional[int] = None
    dtype: Optional[str] = None
    trust_remote_code: bool = True
    enabled: bool = True
    notes: str = ""

    def fingerprint(self) -> Dict[str, Any]:
        """The subset of settings that changes embedding values.

        Deliberately excludes device and batch size so a cache built on a
        laptop is reusable on the cluster.
        """
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "task": self.task,
            "prompt_name": self.prompt_name,
            "prompt_prefix": self.prompt_prefix,
            "modality": self.modality,
            "truncate_dim": self.truncate_dim,
            "pooling": self.pooling,
            "max_length": self.max_length,
            "dtype": self.dtype,
        }


@dataclass
class MetricsConfig:
    bootstrap: int = 0          # 0 disables; 1000 gives the pair-level 95% CI
    bootstrap_seed: int = 12345
    confidence: float = 0.95
    report_domains: bool = True
    #: Datasets entering the unweighted macro average. Overlapping splits
    #: (sick-all vs sick-test) must not both be listed.
    macro_average_over: List[str] = field(
        default_factory=lambda: ["cxc-val", "cxc-test", "sick-test", "sts3k"]
    )


@dataclass
class ExperimentConfig:
    name: str = "text_only_baseline"
    output_dir: Path = REPO_ROOT / "outputs" / "text_only"
    datasets: List[str] = field(default_factory=lambda: ["cxc-val", "cxc-test", "sick-test", "sts3k"])
    limit: Optional[int] = None
    resume: bool = True
    models: List[ModelSpec] = field(default_factory=list)
    data: DataPaths = field(default_factory=DataPaths)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    #: Absolute path of the YAML this was loaded from (recorded in the manifest).
    source_path: Optional[Path] = None
    #: Suffix for manifest/metrics/summary filenames. One TSUBAME job per model
    #: sets this to the model key so parallel jobs sharing an output directory
    #: never overwrite each other's manifest; `sts.cli report` then merges them.
    run_tag: str = ""

    def tagged(self, stem: str, suffix: str) -> Path:
        name = f"{stem}__{self.run_tag}{suffix}" if self.run_tag else f"{stem}{suffix}"
        return self.output_dir / name

    @property
    def enabled_models(self) -> List[ModelSpec]:
        return [m for m in self.models if m.enabled]

    def model_by_key(self, key: str) -> ModelSpec:
        for model in self.models:
            if model.key == key:
                return model
        raise KeyError(f"no model with key {key!r}; have {[m.key for m in self.models]}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        return _stringify_paths(payload)


def _stringify_paths(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _stringify_paths(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_paths(v) for v in obj]
    return obj


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> ExperimentConfig:
    """Parse a YAML experiment file into an :class:`ExperimentConfig`.

    ``overrides`` holds CLI flags that beat the file (e.g. ``limit``,
    ``device``); ``None`` values are ignored so unset flags change nothing.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    base = path.parent.parent if path.parent.name == "configs" else path.parent

    experiment = raw.get("experiment", {}) or {}
    data_raw = raw.get("data", {}) or {}
    runtime_raw = raw.get("runtime", {}) or {}
    metrics_raw = raw.get("metrics", {}) or {}
    models_raw = raw.get("models", []) or []

    data = DataPaths(
        root=_expand(data_raw.get("root", REPO_ROOT / "data" / "raw"), base),
        sick_path=_expand(data_raw["sick_path"], base) if data_raw.get("sick_path") else None,
        cxc_dir=_expand(data_raw["cxc_dir"], base) if data_raw.get("cxc_dir") else None,
        coco_karpathy_json=(
            _expand(data_raw["coco_karpathy_json"], base) if data_raw.get("coco_karpathy_json") else None
        ),
        sts3k_path=_expand(data_raw["sts3k_path"], base) if data_raw.get("sts3k_path") else None,
    )

    runtime_kwargs = dict(runtime_raw)
    if "cache_dir" in runtime_kwargs:
        runtime_kwargs["cache_dir"] = _expand(runtime_kwargs["cache_dir"], base)
    runtime = RuntimeConfig(**runtime_kwargs)

    metrics = MetricsConfig(**metrics_raw)

    models: List[ModelSpec] = []
    for entry in models_raw:
        entry = dict(entry)
        entry.setdefault("key", entry.get("model_id", "model").split("/")[-1])
        unknown = set(entry) - set(ModelSpec.__dataclass_fields__)
        if unknown:
            raise ValueError(f"model {entry['key']!r}: unknown keys {sorted(unknown)}")
        models.append(ModelSpec(**entry))
    if not models:
        raise ValueError(f"{path}: no models defined")

    config = ExperimentConfig(
        name=experiment.get("name", "text_only_baseline"),
        output_dir=_expand(experiment.get("output_dir", REPO_ROOT / "outputs" / "text_only"), base),
        datasets=list(raw.get("datasets", ExperimentConfig().datasets)),
        limit=experiment.get("limit"),
        resume=bool(experiment.get("resume", True)),
        models=models,
        data=data,
        runtime=runtime,
        metrics=metrics,
        source_path=path,
    )

    if "seed" in experiment:
        config.runtime = replace(config.runtime, seed=int(experiment["seed"]))

    for field_name, value in (overrides or {}).items():
        if value is None:
            continue
        if field_name in ("device", "dtype", "batch_size", "use_cache", "cache_dir", "progress"):
            config.runtime = replace(config.runtime, **{field_name: value})
        elif field_name == "bootstrap":
            config.metrics = replace(config.metrics, bootstrap=int(value))
        elif field_name == "output_dir":
            config.output_dir = Path(value)
        elif field_name == "run_tag":
            config.run_tag = str(value)
        elif field_name == "data_root":
            # Re-derive every dataset path from the new root, dropping any
            # per-file paths the YAML set relative to the old one.
            config.data = DataPaths(root=Path(value))
        elif field_name == "models":
            wanted = set(value)
            missing = wanted - {m.key for m in config.models}
            if missing:
                raise ValueError(f"--models: no such model key(s) {sorted(missing)}")
            for model in config.models:
                model.enabled = model.key in wanted
        elif hasattr(config, field_name):
            setattr(config, field_name, value)
        else:
            raise ValueError(f"unknown override {field_name!r}")

    _log.debug("Loaded config %s: %d datasets, %d enabled models",
               path.name, len(config.datasets), len(config.enabled_models))
    return config
