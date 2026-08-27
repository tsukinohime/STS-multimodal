"""Stable content hashes used for reproducibility and cache keys.

Everything here must be deterministic across machines and Python runs, so
``hash()`` (salted per-process) is never used — only hashlib over UTF-8 bytes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Union


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha1_text(text: str) -> str:
    """Short digest for embedding-cache keys (collision risk is negligible here)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file, so multi-hundred-MB datasets stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_obj(obj: Any) -> str:
    """Hash of any JSON-serialisable object with key order normalised."""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return sha256_text(payload)


def slugify(name: str) -> str:
    """Filesystem-safe version of a model id / dataset name."""
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
