"""Content-addressed embedding cache.

Encoding dominates the runtime and the corpora overlap heavily — CxC reuses the
same 25,000 COCO captions across ~44,000 pairs per split, and val/test share
none of them but SICK's two splits do. Caching by *text content* rather than by
pair means all of that is paid for once.

Layout::

    <cache_dir>/<model_key>__<fingerprint>/
        meta.json          # what these vectors are, for a human and for a guard
        shard-00000.npz    # keys: (n,) <U40 sha1 hex, vectors: (n, d) float32
        shard-00001.npz

Shards are append-only and written atomically (temp file + ``os.replace``), so
a job killed by a walltime limit leaves the cache consistent and the next run
resumes from the last flush. The fingerprint covers every setting that changes
a vector (model id, task, prompt, max_length, truncate_dim, dtype) but *not*
device or batch size, so a cache warmed on a laptop is valid on the cluster.
"""

from __future__ import annotations

import json
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import get_logger
from .hashing import sha1_text, sha256_obj, slugify

_log = get_logger()


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {"hits": self.hits, "misses": self.misses, "hit_rate": round(self.hit_rate, 4)}


class EmbeddingCache:
    """Persistent {text -> embedding} store for one encoder configuration."""

    def __init__(
        self,
        cache_dir: Path,
        model_key: str,
        fingerprint: dict,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        self.fingerprint = fingerprint
        self.fingerprint_hash = sha256_obj(fingerprint)[:16]
        self.directory = Path(cache_dir) / f"{slugify(model_key)}__{self.fingerprint_hash}"
        self.stats = CacheStats()

        self._index: Dict[str, int] = {}
        self._vectors: Optional[np.ndarray] = None
        self._pending_keys: List[str] = []
        self._pending_vectors: List[np.ndarray] = []
        self._shard_counter = 0

        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._write_meta()
            self._load_shards()

    # -- persistence -------------------------------------------------------
    def _write_meta(self) -> None:
        meta_path = self.directory / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps({"fingerprint": self.fingerprint, "hash": self.fingerprint_hash}, indent=2),
                encoding="utf-8",
            )

    def _load_shards(self) -> None:
        # Leftover temp files mean a previous process died mid-flush; those
        # vectors were never committed, so drop them.
        for stale in self.directory.glob(".tmp-*.npz"):
            stale.unlink(missing_ok=True)

        shards = sorted(self.directory.glob("shard-*.npz"))
        if not shards:
            return
        keys: List[np.ndarray] = []
        vectors: List[np.ndarray] = []
        for shard in shards:
            try:
                with np.load(shard, allow_pickle=False) as data:
                    keys.append(data["keys"])
                    vectors.append(data["vectors"])
            except (OSError, EOFError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                # A shard truncated by a hard kill: drop it and recompute those
                # texts rather than failing the whole run.
                _log.warning("Discarding unreadable cache shard %s (%s)", shard.name, exc)
                shard.unlink(missing_ok=True)
                continue
        if not keys:
            return
        self._vectors = np.concatenate(vectors, axis=0)
        all_keys = np.concatenate(keys, axis=0)
        self._index = {str(k): i for i, k in enumerate(all_keys)}
        self._shard_counter = len(shards)
        _log.info("Embedding cache %s: %d vectors loaded", self.directory.name, len(self._index))

    def flush(self) -> None:
        """Persist buffered vectors as one new shard, atomically."""
        if not self.enabled or not self._pending_keys:
            return
        keys = np.array(self._pending_keys, dtype="<U40")
        vectors = np.stack(self._pending_vectors).astype(np.float32)

        while (self.directory / f"shard-{self._shard_counter:05d}.npz").exists():
            self._shard_counter += 1
        target = self.directory / f"shard-{self._shard_counter:05d}.npz"

        # The temp name must end in .npz: np.savez silently appends that suffix
        # otherwise, and the rename would then move the wrong (empty) file.
        temp = self.directory / f".tmp-{uuid.uuid4().hex}.npz"
        try:
            with open(temp, "wb") as handle:
                np.savez(handle, keys=keys, vectors=vectors)
            os.replace(temp, target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise

        self._shard_counter += 1
        self._merge_into_memory(keys, vectors)
        self._pending_keys.clear()
        self._pending_vectors.clear()

    def _merge_into_memory(self, keys: np.ndarray, vectors: np.ndarray) -> None:
        start = 0 if self._vectors is None else self._vectors.shape[0]
        self._vectors = vectors if self._vectors is None else np.concatenate([self._vectors, vectors])
        for offset, key in enumerate(keys):
            self._index.setdefault(str(key), start + offset)

    # -- lookup ------------------------------------------------------------
    def lookup(self, texts: Sequence[str]) -> Tuple[Optional[np.ndarray], List[int]]:
        """Return ``(matrix_or_None, missing_positions)`` for ``texts``.

        Rows for missing texts are left as zeros; the caller fills them in via
        :meth:`store` and reads the matrix back afterwards.
        """
        if not self.enabled or self._vectors is None:
            self.stats.misses += len(texts)
            return None, list(range(len(texts)))

        dim = self._vectors.shape[1]
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        missing: List[int] = []
        for position, text in enumerate(texts):
            row = self._index.get(sha1_text(text))
            if row is None:
                missing.append(position)
            else:
                matrix[position] = self._vectors[row]
        self.stats.hits += len(texts) - len(missing)
        self.stats.misses += len(missing)
        return matrix, missing

    def store(self, texts: Sequence[str], vectors: np.ndarray) -> None:
        """Buffer newly computed vectors (call :meth:`flush` to persist)."""
        if not self.enabled:
            return
        for text, vector in zip(texts, vectors):
            key = sha1_text(text)
            if key in self._index:
                continue
            self._pending_keys.append(key)
            self._pending_vectors.append(np.asarray(vector, dtype=np.float32))

    @property
    def pending(self) -> int:
        return len(self._pending_keys)

    def __len__(self) -> int:
        return len(self._index) + len(self._pending_keys)
