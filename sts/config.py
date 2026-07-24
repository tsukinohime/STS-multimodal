"""Configuration and small runtime helpers (paths, device resolution)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Repo root = parent of the `sts` package directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "raw"


def resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to the best available torch device (cuda > mps > cpu)."""
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


def get_logger(name: str = "sts") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@dataclass
class Config:
    """All knobs for a text-only STS evaluation run.

    Paths default to this repository's ``data/raw`` layout, which was verified
    to contain: ``sick/SICK.txt``, ``cxc/data/sts_{val,test}.csv``, and
    ``coco_karpathy/dataset_coco.json``.
    """

    # ---- Data paths ----
    data_root: Path = DEFAULT_DATA_ROOT
    sick_path: Optional[Path] = None            # -> data_root/sick/SICK.txt
    cxc_data_dir: Optional[Path] = None         # -> data_root/cxc/data
    coco_karpathy_json: Optional[Path] = None   # -> data_root/coco_karpathy/dataset_coco.json

    # ---- Encoder ----
    encoder_name: str = "sentence-transformers/all-mpnet-base-v2"
    encoder_backend: str = "sentence-transformers"  # or "hf-mean"
    batch_size: int = 64
    device: str = "auto"
    normalize_embeddings: bool = True
    prompt_prefix: str = ""              # e.g. "query: " for E5, "" for MPNet/MiniLM
    max_seq_length: Optional[int] = None
    hf_pooling: str = "mean"             # for encoder_backend == "hf-mean": "mean" | "cls"

    # ---- Evaluation ----
    sick_split: str = "test"            # all | train | trial | test
    limit: Optional[int] = None          # subsample N pairs per dataset (smoke tests)
    seed: int = 0

    # ---- Output ----
    output_dir: Path = REPO_ROOT / "outputs"

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root)
        self.sick_path = Path(self.sick_path) if self.sick_path else self.data_root / "sick" / "SICK.txt"
        self.cxc_data_dir = Path(self.cxc_data_dir) if self.cxc_data_dir else self.data_root / "cxc" / "data"
        self.coco_karpathy_json = (
            Path(self.coco_karpathy_json)
            if self.coco_karpathy_json
            else self.data_root / "coco_karpathy" / "dataset_coco.json"
        )
        self.output_dir = Path(self.output_dir)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
