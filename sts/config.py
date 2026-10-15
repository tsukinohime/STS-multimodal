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


#: What a model or judge is shown for each sentence: the sentence itself, the
#: image the configured generator made from it (condition D1), or both (D2).
INPUT_KINDS = ("text", "image", "text+image")
#: An encoder embeds one modality per sentence; only a judge takes both.
MODEL_INPUT_KINDS = ("text", "image")


def reads_images(input_kind: str) -> bool:
    """True when the input includes the generated images."""
    return input_kind in ("image", "text+image")


def _input_fingerprint(input_kind: str, image_size: Optional[int]) -> Dict[str, Any]:
    """Extra fingerprint keys for image input.

    Empty for text input on purpose: adding keys would change the hash of every
    existing text cache and silently force a full recompute.
    """
    if input_kind == "text":
        return {}
    return {"input": input_kind, "image_size": image_size}


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
    input: str = "text"               # text | image (the generated image of each sentence)
    image_size: Optional[int] = None  # image input: longer side in pixels; None = as generated
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
            **_input_fingerprint(self.input, self.image_size),
        }


@dataclass
class JudgeSpec:
    """One frozen chat LLM used as a pairwise similarity rater (LLM-as-judge).

    The judge never generates text: the prediction is the probability-weighted
    expectation over the allowed digit tokens at the answer position. See
    :mod:`sts.judge`.
    """

    key: str
    model_id: str
    backend: str = "hf"                 # only the transformers backend for now
    prompt_set: str = "v1"              # which sts.judge.prompts.PROMPT_SETS entry
    enable_thinking: bool = False       # must stay False: no chain-of-thought
    symmetrize: bool = False            # also score (b, a) and average — doubles cost
    symmetry_check_sample: int = 256    # pairs re-scored in swapped order as a diagnostic
    min_valid_mass: float = 0.05        # below this digit-token mass the answer is degenerate (NaN)
    batch_size: Optional[int] = None    # overrides RuntimeConfig.batch_size
    max_prompt_tokens: int = 1024       # flagged, never truncated
    dtype: Optional[str] = None
    trust_remote_code: bool = False
    input: str = "text"                 # text | image (the generated images instead) | text+image (both)
    image_size: Optional[int] = None    # image input: longer side in pixels; None = as generated
    enabled: bool = True
    notes: str = ""

    def fingerprint(self) -> Dict[str, Any]:
        """Settings that change a judgment; the prompt hash is added at run time."""
        return {
            "model_id": self.model_id,
            "backend": self.backend,
            "prompt_set": self.prompt_set,
            "enable_thinking": self.enable_thinking,
            "dtype": self.dtype,
            **_input_fingerprint(self.input, self.image_size),
        }


@dataclass
class GeneratorSpec:
    """The text-to-image model that turns every sentence into one image.

    Every setting that can change a pixel is in :meth:`fingerprint`, which
    names the image-store directory: changing one writes a new store instead of
    mixing two generators' images. ``dtype`` is used exactly as written (never
    resolved from the device), because the jobs that *read* the store run on
    other devices and must arrive at the same directory.
    """

    key: str = "flux2-klein-4b"
    model_id: str = "black-forest-labs/FLUX.2-klein-4B"
    backend: str = "flux2-klein"        # flux2-klein | render-text (laptop stand-in, no model)
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 4        # klein is step-distilled to 4 steps
    guidance_scale: float = 1.0         # the distilled checkpoint ignores guidance; recorded anyway
    max_sequence_length: int = 512      # prompt tokens; longer prompts are counted and reported
    seed: int = 0
    seed_policy: str = "shared"         # shared: one seed for every sentence | per-text: hash of the text
    dtype: str = "bfloat16"
    batch_size: int = 8                 # prompts per pipeline call
    image_format: str = "jpeg"          # jpeg | png
    jpeg_quality: int = 95
    notes: str = ""

    def fingerprint(self) -> Dict[str, Any]:
        fingerprint = {
            # One local directory spelled two ways (a trailing slash, say) must
            # name one store; normpath leaves a Hub id as it is.
            "model_id": os.path.normpath(self.model_id),
            "backend": self.backend,
            "height": self.height,
            "width": self.width,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "max_sequence_length": self.max_sequence_length,
            "seed": self.seed,
            "seed_policy": self.seed_policy,
            "dtype": self.dtype,
            "image_format": self.image_format,
        }
        if self.image_format == "jpeg":
            fingerprint["jpeg_quality"] = self.jpeg_quality
        return fingerprint


@dataclass
class ImagesConfig:
    """Where generated images live and how they are produced (conditions D*, E*)."""

    store_dir: Path = REPO_ROOT / ".cache" / "images"
    #: Images per tar shard: the resume granularity and the file-count lever.
    flush_every: int = 512
    #: Pairs per dataset rendered side by side into <output_dir>/images_preview/.
    preview_pairs: int = 4
    generator: Optional[GeneratorSpec] = None

    def __post_init__(self) -> None:
        self.store_dir = Path(self.store_dir)


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
    judges: List[JudgeSpec] = field(default_factory=list)
    data: DataPaths = field(default_factory=DataPaths)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    images: ImagesConfig = field(default_factory=ImagesConfig)
    #: Experimental condition this config implements (e.g. "D1"); selects the
    #: method paragraph of the summary. Empty for the text-only baselines.
    condition: str = ""
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

    @property
    def enabled_judges(self) -> List[JudgeSpec]:
        return [j for j in self.judges if j.enabled]

    @property
    def uses_images(self) -> bool:
        """True when an enabled model or judge reads generated images."""
        return any(reads_images(s.input) for s in [*self.enabled_models, *self.enabled_judges])

    def judge_by_key(self, key: str) -> JudgeSpec:
        for judge in self.judges:
            if judge.key == key:
                return judge
        raise KeyError(f"no judge with key {key!r}; have {[j.key for j in self.judges]}")

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
    judges_raw = raw.get("judges", []) or []
    images_raw = dict(raw.get("images", {}) or {})

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

    judges: List[JudgeSpec] = []
    for entry in judges_raw:
        entry = dict(entry)
        entry.setdefault("key", entry.get("model_id", "judge").split("/")[-1])
        unknown = set(entry) - set(JudgeSpec.__dataclass_fields__)
        if unknown:
            raise ValueError(f"judge {entry['key']!r}: unknown keys {sorted(unknown)}")
        judges.append(JudgeSpec(**entry))

    if not models and not judges:
        raise ValueError(f"{path}: no models or judges defined")
    clash = {m.key for m in models} & {j.key for j in judges}
    if clash:
        raise ValueError(f"{path}: keys used by both a model and a judge: {sorted(clash)}")
    for spec, kinds in [*((m, MODEL_INPUT_KINDS) for m in models), *((j, INPUT_KINDS) for j in judges)]:
        if spec.input not in kinds:
            raise ValueError(f"{spec.key!r}: input must be one of {kinds}, got {spec.input!r}")

    generator_raw = images_raw.pop("generator", None)
    generator = None
    if generator_raw:
        unknown = set(generator_raw) - set(GeneratorSpec.__dataclass_fields__)
        if unknown:
            raise ValueError(f"images.generator: unknown keys {sorted(unknown)}")
        generator = GeneratorSpec(**generator_raw)
    unknown = set(images_raw) - set(ImagesConfig.__dataclass_fields__)
    if unknown:
        raise ValueError(f"images: unknown keys {sorted(unknown)}")
    if "store_dir" in images_raw:
        images_raw["store_dir"] = _expand(images_raw["store_dir"], base)
    images = ImagesConfig(**images_raw, generator=generator)
    needs_images = [s.key for s in [*models, *judges] if reads_images(s.input)]
    if needs_images and generator is None:
        raise ValueError(f"{path}: {needs_images} read generated images, but images.generator is not set")

    config = ExperimentConfig(
        name=experiment.get("name", "text_only_baseline"),
        output_dir=_expand(experiment.get("output_dir", REPO_ROOT / "outputs" / "text_only"), base),
        datasets=list(raw.get("datasets", ExperimentConfig().datasets)),
        limit=experiment.get("limit"),
        resume=bool(experiment.get("resume", True)),
        models=models,
        judges=judges,
        data=data,
        runtime=runtime,
        metrics=metrics,
        images=images,
        condition=str(experiment.get("condition", "") or ""),
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
        elif field_name == "image_dir":
            config.images = replace(config.images, store_dir=Path(value))
        elif field_name == "run_tag":
            config.run_tag = str(value)
        elif field_name == "data_root":
            # Re-derive every dataset path from the new root, dropping any
            # per-file paths the YAML set relative to the old one.
            config.data = DataPaths(root=Path(value))
        elif field_name == "models":
            # One flag selects among encoders and judges alike.
            wanted = set(value)
            known = {m.key for m in config.models} | {j.key for j in config.judges}
            missing = wanted - known
            if missing:
                raise ValueError(f"--models: no such model/judge key(s) {sorted(missing)}")
            for model in config.models:
                model.enabled = model.key in wanted
            for judge in config.judges:
                judge.enabled = judge.key in wanted
        elif hasattr(config, field_name):
            setattr(config, field_name, value)
        else:
            raise ValueError(f"unknown override {field_name!r}")

    _log.debug("Loaded config %s: %d datasets, %d enabled models",
               path.name, len(config.datasets), len(config.enabled_models))
    return config
