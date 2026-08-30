"""Experiment manifest: everything needed to reproduce or audit a run.

Captures the resolved config (and its hash), library and hardware versions, the
exact model revisions, and a SHA-256 per data file. Written incrementally so a
job killed mid-run still leaves a readable record of what it did.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from .config import ExperimentConfig, get_logger, resolve_device, resolve_dtype
from .hashing import sha256_obj

_log = get_logger()


def _run_git(*args: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parents[1],
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def git_state() -> Dict[str, Any]:
    dirty = _run_git("status", "--porcelain")
    return {
        "commit": _run_git("rev-parse", "HEAD"),
        "branch": _run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
    }


def package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "sentence_transformers", "peft",
                 "numpy", "scipy", "pandas", "huggingface_hub"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            versions[name] = None
    return versions


def hardware_info(device: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "device": device,
        "hostname": platform.node(),
    }
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            info["gpu_name"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 1024 ** 3, 1)
            info["gpu_count"] = torch.cuda.device_count()
            info["cuda_version"] = torch.version.cuda
        elif device == "mps":
            info["gpu_name"] = "Apple Silicon (MPS)"
    except Exception:  # torch missing or a driver hiccup must not kill the run
        pass
    # Grid Engine fills these on TSUBAME; absent locally. Recorded so a result
    # can be traced back to the exact scheduler job that produced it.
    for key in ("JOB_ID", "JOB_NAME", "QUEUE", "SGE_O_HOST", "NSLOTS",
                "PE_HOSTFILE", "SGE_TASK_ID"):
        if os.environ.get(key):
            info[f"sge_{key.lower()}"] = os.environ[key]
    return info


def resolve_model_revision(model_id: str) -> Optional[str]:
    """Best-effort HF commit sha, working offline.

    Reads the local hub cache ref first so an offline compute node still gets a
    pinned revision; only falls back to the Hub API when the cache is silent.
    """
    try:
        from huggingface_hub import constants

        cache_root = Path(os.environ.get("HF_HUB_CACHE") or constants.HF_HUB_CACHE)
        ref = cache_root / f"models--{model_id.replace('/', '--')}" / "refs" / "main"
        if ref.exists():
            return ref.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return None
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id).sha
    except Exception:
        return None


@dataclass
class Manifest:
    config: ExperimentConfig
    path: Path
    started_at: float = field(default_factory=time.time)
    payload: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        device = resolve_device(self.config.runtime.device)
        dtype = resolve_dtype(self.config.runtime.dtype, device)
        config_dict = self.config.to_dict()
        self.payload = {
            "experiment": self.config.name,
            "run_started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(self.started_at)),
            "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
            "config_file": str(self.config.source_path) if self.config.source_path else None,
            "config": config_dict,
            "config_sha256": sha256_obj(config_dict),
            "seed": self.config.runtime.seed,
            "resolved": {
                "device": device,
                "dtype": dtype,
                "batch_size": self.config.runtime.batch_size,
                "max_length": self.config.runtime.max_length,
                "cache_dir": str(self.config.runtime.cache_dir),
                "use_cache": self.config.runtime.use_cache,
            },
            "git": git_state(),
            "versions": package_versions(),
            "hardware": hardware_info(device),
            "env": {
                key: os.environ.get(key)
                for key in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE",
                            "TRANSFORMERS_OFFLINE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS")
                if os.environ.get(key)
            },
            "models": {},
            "datasets": {},
            "runs": [],
        }

    def record_model(self, key: str, model_id: str, details: Dict[str, Any]) -> None:
        self.payload["models"][key] = {
            "model_id": model_id,
            "revision": resolve_model_revision(model_id),
            **details,
        }
        self.save()

    def record_dataset(self, dataset) -> None:
        self.payload["datasets"][dataset.name] = {
            "split": dataset.split,
            "n_pairs": len(dataset),
            "n_rows_in_source": dataset.n_rows_in_source,
            "score_range": list(dataset.score_range) if dataset.score_range else None,
            "source_files": dataset.source_files,
            "domains": sorted({p.domain for p in dataset.pairs}),
            "notes": dataset.notes,
        }
        self.save()

    def record_run(self, entry: Dict[str, Any]) -> None:
        self.payload["runs"].append(entry)
        self.save()

    def finish(self) -> None:
        self.payload["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.payload["total_runtime_seconds"] = round(time.time() - self.started_at, 2)
        self.save()
        _log.info("Manifest -> %s", self.path)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(self.payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self.path)
