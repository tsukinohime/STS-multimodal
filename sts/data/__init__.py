"""Dataset adapters: native corpus formats -> the uniform :class:`STSDataset`."""

from .base import PAIR_COLUMNS, STSDataset, STSPair, clean_text
from .cxc import CaptionMappingError, load_coco_karpathy_index, load_cxc_sts
from .registry import AVAILABLE_DATASETS, load_dataset
from .sick import load_sick
from .sts3k import load_sts3k

__all__ = [
    "PAIR_COLUMNS",
    "STSDataset",
    "STSPair",
    "clean_text",
    "CaptionMappingError",
    "load_coco_karpathy_index",
    "load_cxc_sts",
    "load_sick",
    "load_sts3k",
    "AVAILABLE_DATASETS",
    "load_dataset",
]
