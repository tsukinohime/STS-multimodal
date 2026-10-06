"""Content-addressed store for generated images: one image per (sentence, generator setting).

Every image condition (D1 image only, D2 image + text, E2 mismatched images,
...) reads the same images, so they are generated once and stored under the
text they came from::

    <store_dir>/<generator_key>__<fingerprint>/
        meta.json                            # generator settings, for a human and as a guard
        shard-20261005T120000-1a2b3c4d.tar   # uncompressed tar: <sha1>.jpg ... + index.json
        shard-20261005T121502-5e6f7a8b.tar

* **Sharded tar, not one file per image.** ~60k images would cost ~60k inodes
  on a cluster filesystem that meters them; shards of 512 cost ~120. An
  uncompressed tar keeps every member readable at a fixed offset, so a lookup
  is one positioned read, and ``tar -xf`` gets any image back for a look.
* **Content-addressed.** A member is named by the SHA-1 of its sentence and
  the directory by a hash of every generator setting, so a stored image is
  reused exactly when the same sentence meets the same generator.
* **Append-only and atomic.** A shard is written to a temp file and renamed
  into place, so a job killed mid-write leaves the store consistent. Shard
  names carry a timestamp and a random suffix rather than a counter, so
  several generation jobs (e.g. one per dataset) can fill one store at once.
* **Thread-safe reads** through ``os.pread`` on shared descriptors, so a loader
  thread can decode the next batch while the GPU works on this one.

``index.json`` inside each shard maps every member back to its sentence and
seed — the SHA-1 alone cannot be inverted.
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, TypeVar

from ..config import ExperimentConfig, get_logger
from ..hashing import sha1_text, sha256_obj, slugify

if TYPE_CHECKING:  # pragma: no cover
    from PIL.Image import Image

_log = get_logger()

T = TypeVar("T")
R = TypeVar("R")

_EXTENSIONS = {"jpeg": ".jpg", "png": ".png"}
_INDEX_MEMBER = "index.json"
#: A temp shard older than this belongs to a dead writer: a live flush takes seconds.
_STALE_TEMP_SECONDS = 3600
#: Processors of the Qwen-VL family (jina-v5-omni, Qwen3.5) cut images into 16 px
#: patches merged 2 x 2, so sides that are multiples of 32 pass through unresized.
VISION_MULTIPLE = 32


class MissingImagesError(FileNotFoundError):
    """Sentences that have no generated image in the store."""


@dataclass(frozen=True)
class _Location:
    shard: Path
    offset: int
    size: int


def fit_image(image: "Image", size: Optional[int], multiple: int = VISION_MULTIPLE) -> "Image":
    """Resize so the longer side is ``size`` (aspect kept), each side a multiple of ``multiple``.

    ``None`` returns the image untouched. Bicubic, the resampling the Qwen-VL
    image processors use themselves.
    """
    if not size:
        return image
    from PIL import Image as PILImage

    width, height = image.size
    scale = size / max(width, height)
    new_width = max(multiple, round(width * scale / multiple) * multiple)
    new_height = max(multiple, round(height * scale / multiple) * multiple)
    if (new_width, new_height) == (width, height):
        return image
    return image.resize((new_width, new_height), PILImage.Resampling.BICUBIC)


def visual_tokens(width: int, height: int, size: Optional[int] = None) -> int:
    """Image tokens a Qwen-VL-family processor makes of this image after :func:`fit_image`."""
    if size:
        scale = size / max(width, height)
        width = max(VISION_MULTIPLE, round(width * scale / VISION_MULTIPLE) * VISION_MULTIPLE)
        height = max(VISION_MULTIPLE, round(height * scale / VISION_MULTIPLE) * VISION_MULTIPLE)
    return (width // VISION_MULTIPLE) * (height // VISION_MULTIPLE)


def iter_prefetched(items: Sequence[T], load: Callable[[T], R]) -> Iterator[R]:
    """``load(item)`` for every item in order, the next one loading in a background thread.

    Exactly one item is loaded ahead, so at most two batches of decoded images
    are in memory while the GPU works on one of them.
    """
    if not items:
        return
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load, items[0])
        for index in range(len(items)):
            result = future.result()
            if index + 1 < len(items):
                future = pool.submit(load, items[index + 1])
            yield result


class ImageStore:
    """Persistent {sentence -> generated image} store for one generator setting."""

    def __init__(
        self,
        root: Path,
        generator_key: str,
        fingerprint: dict,
        image_format: str = "jpeg",
        jpeg_quality: int = 95,
        create: bool = True,
    ) -> None:
        if image_format not in _EXTENSIONS:
            raise ValueError(f"image_format must be one of {sorted(_EXTENSIONS)}, got {image_format!r}")
        self.generator_key = generator_key
        self.fingerprint = dict(fingerprint)
        self.fingerprint_hash = sha256_obj(self.fingerprint)[:16]
        self.directory = Path(root) / f"{slugify(generator_key)}__{self.fingerprint_hash}"
        self.image_format = image_format
        self.jpeg_quality = jpeg_quality
        self.extension = _EXTENSIONS[image_format]

        self._index: Dict[str, _Location] = {}
        self._entries: Dict[str, dict] = {}
        self._pending: List[Tuple[str, dict, Future]] = []
        self._pending_keys: set = set()
        self._encoder: Optional[ThreadPoolExecutor] = None
        self._fds: Dict[Path, int] = {}
        self._fd_lock = threading.Lock()
        self.n_shards = 0

        if create:
            self.directory.mkdir(parents=True, exist_ok=True)
            self._write_meta()
        if self.directory.is_dir():
            self._load_shards()

    # -- persistence -------------------------------------------------------
    def _write_meta(self) -> None:
        meta_path = self.directory / "meta.json"
        if meta_path.exists():
            return
        temp = self.directory / f".tmp-{uuid.uuid4().hex}.json"
        temp.write_text(
            json.dumps({"generator": self.generator_key, "fingerprint": self.fingerprint,
                        "hash": self.fingerprint_hash}, indent=2),
            encoding="utf-8",
        )
        os.replace(temp, meta_path)

    def _load_shards(self) -> None:
        now = time.time()
        for stale in self.directory.glob(".tmp-*"):
            try:
                if now - stale.stat().st_mtime > _STALE_TEMP_SECONDS:
                    stale.unlink()
            except FileNotFoundError:
                pass

        shards = sorted(self.directory.glob("shard-*.tar"))
        for shard in shards:
            try:
                self._index_shard(shard)
            except (tarfile.TarError, OSError, ValueError, KeyError) as exc:
                _log.warning("Skipping unreadable image shard %s (%s)", shard.name, exc)
        self.n_shards = len(shards)
        if self._index:
            _log.info("Image store %s: %d images in %d shard(s)",
                      self.directory.name, len(self._index), len(shards))

    def _index_shard(self, shard: Path) -> None:
        entries: Dict[str, dict] = {}
        with tarfile.open(shard, mode="r:") as tar:
            members = tar.getmembers()
            for member in members:
                if member.name == _INDEX_MEMBER:
                    handle = tar.extractfile(member)
                    if handle is not None:
                        entries = {entry["key"]: entry for entry in json.load(handle)}
        for member in members:
            if not member.isfile() or member.name == _INDEX_MEMBER:
                continue
            key = member.name.rsplit(".", 1)[0]
            # First shard listing a key wins; sorted() makes that stable. Two
            # concurrent jobs generating the same sentence is harmless.
            if key not in self._index:
                self._index[key] = _Location(shard, member.offset_data, member.size)
                if key in entries:
                    self._entries[key] = entries[key]

    # -- queries -----------------------------------------------------------
    @staticmethod
    def key_of(text: str) -> str:
        return sha1_text(text)

    def __contains__(self, text: str) -> bool:
        return self.key_of(text) in self._index

    def __len__(self) -> int:
        return len(self._index)

    def missing(self, texts: Iterable[str]) -> List[str]:
        """Texts with neither a stored nor a pending image, deduplicated, in order."""
        out: List[str] = []
        seen: set = set()
        for text in texts:
            key = self.key_of(text)
            if key in self._index or key in self._pending_keys or key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    def require(self, texts: Sequence[str], purpose: str = "") -> None:
        """Fail with an actionable message unless every text has a stored image."""
        unique = list(dict.fromkeys(texts))
        missing = [t for t in unique if self.key_of(t) not in self._index]
        if missing:
            where = f" for {purpose}" if purpose else ""
            raise MissingImagesError(
                f"{len(missing)} of {len(unique)} sentence(s) have no generated image{where} in "
                f"{self.directory} (first: {missing[0]!r}). Run `python -m sts.cli imagine` with the "
                "same config, datasets and --limit first; the generator settings (model_id "
                "included) must match too, since they name the store directory."
            )

    def entry(self, text: str) -> Optional[dict]:
        """The index.json record (sentence, seed, size) of a stored image."""
        return self._entries.get(self.key_of(text))

    # -- writing -----------------------------------------------------------
    def add(self, text: str, image: "Image", meta: Optional[dict] = None) -> None:
        """Queue an image; it is encoded in the background and written by :meth:`flush`."""
        key = self.key_of(text)
        if key in self._index or key in self._pending_keys:
            return
        if self._encoder is None:
            self._encoder = ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1))
        entry = {"key": key, "text": text, "width": image.width, "height": image.height, **(meta or {})}
        self._pending.append((key, entry, self._encoder.submit(self._encode, image)))
        self._pending_keys.add(key)

    def _encode(self, image: "Image") -> bytes:
        buffer = io.BytesIO()
        image = image.convert("RGB")
        if self.image_format == "jpeg":
            # 4:4:4 chroma: no colour subsampling on top of the quality setting.
            image.save(buffer, format="JPEG", quality=self.jpeg_quality, subsampling=0)
        else:
            image.save(buffer, format="PNG")
        return buffer.getvalue()

    @property
    def pending(self) -> int:
        return len(self._pending)

    def flush(self) -> Optional[Path]:
        """Write the queued images as one new shard, atomically. Returns its path."""
        if not self._pending:
            return None
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        target = self.directory / f"shard-{stamp}-{uuid.uuid4().hex[:8]}.tar"
        temp = self.directory / f".tmp-{uuid.uuid4().hex}.tar"
        entries: List[dict] = []
        try:
            with tarfile.open(temp, mode="w") as tar:
                for key, entry, future in self._pending:
                    _add_member(tar, key + self.extension, future.result())
                    entries.append(entry)
                _add_member(tar, _INDEX_MEMBER, json.dumps(entries, ensure_ascii=False).encode("utf-8"))
            os.replace(temp, target)
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
        self._pending.clear()
        self._pending_keys.clear()
        self._index_shard(target)
        self.n_shards += 1
        return target

    # -- reading -----------------------------------------------------------
    def _fd(self, shard: Path) -> int:
        with self._fd_lock:
            fd = self._fds.get(shard)
            if fd is None:
                fd = os.open(shard, os.O_RDONLY)
                self._fds[shard] = fd
            return fd

    def read_bytes(self, text: str) -> bytes:
        location = self._index.get(self.key_of(text))
        if location is None:
            raise MissingImagesError(f"no generated image for {text!r} in {self.directory}")
        data = os.pread(self._fd(location.shard), location.size, location.offset)
        if len(data) != location.size:
            raise OSError(f"short read from {location.shard.name}: {len(data)} of {location.size} bytes")
        return data

    def load(self, text: str, size: Optional[int] = None) -> "Image":
        """The decoded RGB image of ``text``, optionally :func:`fit_image`-resized."""
        from PIL import Image as PILImage

        image = PILImage.open(io.BytesIO(self.read_bytes(text))).convert("RGB")
        return fit_image(image, size)

    # -- bookkeeping -------------------------------------------------------
    def stat(self) -> Dict[str, object]:
        shards = sorted(self.directory.glob("shard-*.tar")) if self.directory.is_dir() else []
        total = sum(s.stat().st_size for s in shards)
        return {
            "directory": str(self.directory),
            "n_images": len(self._index),
            "n_shards": len(shards),
            "bytes": total,
            "mean_image_bytes": int(total / len(self._index)) if self._index else 0,
        }

    def describe(self) -> Dict[str, object]:
        return {
            "generator": self.generator_key,
            "store": str(self.directory),
            "store_hash": self.fingerprint_hash,
            "fingerprint": self.fingerprint,
        }

    def close(self) -> None:
        """Flush nothing, release descriptors and the encoder pool. Safe to call twice."""
        if self._pending:
            _log.warning("Image store closed with %d unflushed image(s); they are dropped", len(self._pending))
            self._pending.clear()
            self._pending_keys.clear()
        if self._encoder is not None:
            self._encoder.shutdown(wait=True)
            self._encoder = None
        with self._fd_lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


def open_image_store(config: ExperimentConfig, create: bool = False) -> ImageStore:
    """The store of ``config.images.generator``; ``create`` only for the generating stage."""
    spec = config.images.generator
    if spec is None:
        raise ValueError(f"{config.source_path}: image input needs an images.generator section")
    return ImageStore(
        config.images.store_dir, spec.key, spec.fingerprint(),
        image_format=spec.image_format, jpeg_quality=spec.jpeg_quality, create=create,
    )


def _add_member(tar: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = int(time.time())
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(payload))
