"""Transparent read access to data files that live inside ``.tar`` archives.

Cluster filesystems meter inode count as well as bytes, so large corpora are
better stored as one archive than as thousands of small files — TSUBAME's
storage guidance says exactly this, and COCO's ~123k images make it unavoidable
once the visual milestone starts.

This module lets every dataset adapter keep asking for a plain path. Given
``data/raw/coco_karpathy/dataset_coco.json``, it returns the real file if it
exists, and otherwise looks for that member inside an archive sitting in an
ancestor directory (``data/raw/coco.tar`` holds ``coco_karpathy/…``). The
archive's own name need not match the directory it contains, so resolution
works by indexing archive *members*, not by guessing filenames.

Consequences worth knowing:

* The same YAML config works whether the data is extracted or archived.
* Uncompressed ``.tar`` is strongly preferred: members are found by seeking.
  ``.tar.gz`` is accepted but every read decompresses from the start of the
  archive, which is slow and gets slower the deeper the member sits.
* Archives are opened once and cached, so the member index is built once per
  process. Not thread-safe — ``tarfile`` objects never are.
"""

from __future__ import annotations

import atexit
import hashlib
import io
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Tuple, Union

from .config import get_logger

_log = get_logger()

#: Suffixes treated as archives. Order matters for suffix stripping.
ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
#: How far up the tree to look for an archive holding the requested path.
_MAX_WALK_UP = 5

#: archive path -> open TarFile, so the member index is built once per process.
_OPEN_ARCHIVES: Dict[str, tarfile.TarFile] = {}

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Resolution:
    """Where a requested data path actually lives."""

    path: Path                      # what the caller asked for
    archive: Optional[Path] = None  # the .tar holding it, if any
    member: Optional[str] = None    # member name inside that archive

    @property
    def in_archive(self) -> bool:
        return self.archive is not None

    @property
    def display(self) -> str:
        """Human-readable location, recorded in the run manifest."""
        return f"{self.archive}::{self.member}" if self.in_archive else str(self.path)


def _is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _open_archive(archive: Path) -> tarfile.TarFile:
    key = str(archive)
    handle = _OPEN_ARCHIVES.get(key)
    if handle is None:
        if archive.name.lower().endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")):
            _log.warning(
                "%s is compressed; every member read decompresses from the start. "
                "Repack as an uncompressed .tar for random access.", archive.name,
            )
        handle = tarfile.open(archive, mode="r")
        _OPEN_ARCHIVES[key] = handle
    return handle


@lru_cache(maxsize=32)
def _member_index(directory: str) -> Dict[str, Tuple[str, str]]:
    """``{member_path: (archive, member)}`` for every archive in ``directory``.

    Cached per directory: building it scans each archive's header block once.
    """
    index: Dict[str, Tuple[str, str]] = {}
    folder = Path(directory)
    if not folder.is_dir():
        return index

    for entry in sorted(folder.iterdir()):
        if not entry.is_file() or not _is_archive(entry):
            continue
        try:
            handle = _open_archive(entry)
            names = handle.getnames()
        except (tarfile.TarError, OSError) as exc:
            _log.warning("Skipping unreadable archive %s (%s)", entry.name, exc)
            continue
        for name in names:
            # First archive listing a member wins; sorted() makes that stable.
            index.setdefault(name.lstrip("./"), (str(entry), name))
        _log.debug("Indexed %s: %d members", entry.name, len(names))
    return index


def resolve(path: PathLike) -> Resolution:
    """Locate ``path`` on disk, or as a member of a nearby archive."""
    path = Path(path)
    if path.is_file():
        return Resolution(path=path)

    # Walk up looking for a directory whose archives contain this path.
    for depth, parent in enumerate(path.parents):
        if depth >= _MAX_WALK_UP:
            break
        index = _member_index(str(parent))
        if not index:
            continue
        try:
            relative = path.relative_to(parent).as_posix()
        except ValueError:  # pragma: no cover - parents are always prefixes
            continue
        hit = index.get(relative)
        if hit:
            return Resolution(path=path, archive=Path(hit[0]), member=hit[1])

    return Resolution(path=path)


def exists(path: PathLike) -> bool:
    resolution = resolve(path)
    return resolution.path.is_file() or resolution.in_archive


def open_binary(path: PathLike) -> BinaryIO:
    """Open ``path`` for binary reading, from disk or from its archive.

    Archive members are read fully into memory: ``tarfile`` streams are not
    seekable, and pandas/json both want to seek. These files are at most a few
    hundred MB, which is well within a compute node's memory.
    """
    resolution = resolve(path)
    if not resolution.in_archive:
        if not resolution.path.is_file():
            raise FileNotFoundError(_not_found_message(path))
        return open(resolution.path, "rb")

    handle = _open_archive(resolution.archive)
    extracted = handle.extractfile(resolution.member)
    if extracted is None:
        raise FileNotFoundError(
            f"{resolution.member!r} in {resolution.archive} is not a regular file"
        )
    with extracted:
        return io.BytesIO(extracted.read())


def sha256(path: PathLike, chunk_size: int = 1 << 20) -> str:
    """Hash of the file's *content*, identical whether extracted or archived.

    Hashing the member rather than the enclosing archive keeps manifests
    comparable across storage layouts.
    """
    digest = hashlib.sha256()
    with open_binary(path) as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display(path: PathLike) -> str:
    """``/…/coco.tar::coco_karpathy/dataset_coco.json`` or the plain path."""
    return resolve(path).display


def _not_found_message(path: PathLike) -> str:
    path = Path(path)
    searched = []
    for depth, parent in enumerate(path.parents):
        if depth >= _MAX_WALK_UP:
            break
        archives = [e.name for e in parent.iterdir() if e.is_file() and _is_archive(e)] \
            if parent.is_dir() else []
        if archives:
            searched.append(f"{parent} ({', '.join(archives)})")
    hint = f"; archives searched: {'; '.join(searched)}" if searched else ""
    return f"data file not found: {path}{hint}"


def close_all() -> None:
    """Close cached archive handles. Registered at exit; safe to call early."""
    for handle in _OPEN_ARCHIVES.values():
        try:
            handle.close()
        except OSError:
            pass
    _OPEN_ARCHIVES.clear()
    _member_index.cache_clear()


atexit.register(close_all)
