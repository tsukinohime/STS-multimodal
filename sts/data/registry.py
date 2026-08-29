"""Name -> dataset adapter dispatch.

Dataset names in the YAML config resolve here. Keeping the mapping in one place
means a new corpus is one adapter plus one entry.
"""

from __future__ import annotations

from typing import Callable, Dict

from ..config import DataPaths
from .base import STSDataset
from .cxc import load_cxc_sts
from .sick import load_sick
from .sts3k import load_sts3k

#: name -> (loader, one-line description) for `--list-datasets`.
AVAILABLE_DATASETS: Dict[str, str] = {
    "cxc-val": "CxC caption-caption STS, val split (sts_val.csv)",
    "cxc-test": "CxC caption-caption STS, test split (sts_test.csv)",
    "sick-test": "SICK-Relatedness, official TEST split",
    "sick-all": "SICK-Relatedness, every pair (TRAIN+TRIAL+TEST); overlaps sick-test",
    "sts3k": "STS3k, complete file",
}


def load_dataset(name: str, paths: DataPaths) -> STSDataset:
    key = name.strip().lower()
    loaders: Dict[str, Callable[[], STSDataset]] = {
        "cxc-val": lambda: load_cxc_sts(paths.cxc_dir / "sts_val.csv", paths.coco_karpathy_json, "val"),
        "cxc-test": lambda: load_cxc_sts(paths.cxc_dir / "sts_test.csv", paths.coco_karpathy_json, "test"),
        "sick-test": lambda: load_sick(paths.sick_path, split="test"),
        "sick-all": lambda: load_sick(paths.sick_path, split="all"),
        "sick-train": lambda: load_sick(paths.sick_path, split="train"),
        "sick-trial": lambda: load_sick(paths.sick_path, split="trial"),
        "sts3k": lambda: load_sts3k(paths.sts3k_path),
    }
    if key not in loaders:
        raise ValueError(
            f"unknown dataset {name!r}; available: {sorted(AVAILABLE_DATASETS)}"
        )
    return loaders[key]()
