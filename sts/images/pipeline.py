"""Stage 1 of every image condition: one generated image per unique sentence.

    python -m sts.cli imagine --config configs/d1_image_only.yaml

Collects the unique sentences of every configured dataset — after ``--limit``
subsampling with the same seed the scoring commands use, so a smoke run
generates exactly the images its smoke scoring will ask for — skips those the
store already holds, and generates the rest in batches, flushing a shard every
``images.flush_every`` images. A killed job loses at most one shard's worth of
work, and several jobs over different ``--datasets`` can fill one store.

It also writes ``<output_dir>/images_preview/``: a few pairs per dataset, the
two images side by side above their sentences and gold score, so the images
can be looked at before any GPU hour goes into scoring them.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from ..config import ExperimentConfig, GeneratorSpec, get_logger, resolve_device
from ..data import STSDataset, STSPair, load_dataset
from ..manifest import Manifest
from ..pipeline import set_global_seed
from .generators import ImageGenerator, build_generator, font_of_size, seed_for, wrap_text
from .store import ImageStore, open_image_store

_log = get_logger()

#: Side of each image in a preview sheet (a multiple of 32, see ``fit_image``).
_PREVIEW_SIDE = 384


def load_datasets(config: ExperimentConfig) -> Dict[str, STSDataset]:
    """The configured datasets, subsampled exactly as the scoring commands subsample them."""
    datasets: Dict[str, STSDataset] = {}
    for name in config.datasets:
        dataset = load_dataset(name, config.data)
        if config.limit:
            dataset = dataset.subsample(config.limit, seed=config.runtime.seed)
        datasets[name] = dataset
    return datasets


def _generate(
    generator: ImageGenerator,
    store: ImageStore,
    todo: Sequence[str],
    prompt_tokens: np.ndarray,
    spec: GeneratorSpec,
    flush_every: int,
    progress: bool,
) -> int:
    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(total=len(todo), desc=f"imagine[{spec.key}]", unit="img")
        except ImportError:
            pass
    done = 0
    try:
        for start in range(0, len(todo), spec.batch_size):
            texts = list(todo[start : start + spec.batch_size])
            seeds = [seed_for(text, spec) for text in texts]
            images = generator.generate(texts, seeds)
            if len(images) != len(texts):
                raise RuntimeError(f"generator returned {len(images)} images for {len(texts)} prompts")
            for offset, (text, seed, image) in enumerate(zip(texts, seeds, images)):
                store.add(text, image, {"seed": seed, "prompt_tokens": int(prompt_tokens[start + offset])})
            done += len(texts)
            if store.pending >= flush_every:
                store.flush()
            if bar is not None:
                bar.update(len(texts))
    finally:
        if bar is not None:
            bar.close()
    return done


def run_generation(config: ExperimentConfig) -> Dict[str, object]:
    """Generate every missing image for ``config``'s datasets. Returns a run summary."""
    spec = config.images.generator
    if spec is None:
        raise ValueError(f"{config.source_path}: no images.generator section to generate with")
    set_global_seed(config.runtime.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = Manifest(config=config, path=config.tagged("imagine", ".json"))
    manifest.payload["method"] = "image-generation"
    datasets = load_datasets(config)
    for dataset in datasets.values():
        manifest.record_dataset(dataset)
    texts = list(dict.fromkeys(t for dataset in datasets.values() for t in dataset.unique_texts()))

    store = open_image_store(config, create=True)
    todo = store.missing(texts)
    _log.info("Generator %s -> %s | %d unique sentences: %d stored, %d to generate",
              spec.key, store.directory, len(texts), len(texts) - len(todo), len(todo))

    generator = None
    n_long = 0
    started = time.time()
    try:
        if todo:
            generator = build_generator(spec, resolve_device(config.runtime.device))
            manifest.record_model(spec.key, spec.model_id, {**generator.describe(), **store.describe()})
            prompt_tokens = generator.prompt_tokens(todo)
            n_long = int((prompt_tokens > spec.max_sequence_length).sum())
            if n_long:
                _log.warning("%d prompt(s) exceed max_sequence_length=%d tokens; the pipeline truncates them",
                             n_long, spec.max_sequence_length)
            _generate(generator, store, todo, prompt_tokens, spec,
                      config.images.flush_every, config.runtime.progress)
    finally:
        # Whatever finished before an error is a complete image: keep it.
        store.flush()
        if generator is not None:
            generator.close()
    elapsed = time.time() - started

    store.require(texts, purpose=config.name)
    stat = store.stat()
    manifest.record_run({
        "model": spec.key,
        "dataset": "+".join(datasets),
        "n_pairs": sum(len(d) for d in datasets.values()),
        "n_unique_texts": len(texts),
        "n_generated": len(todo),
        "n_reused": len(texts) - len(todo),
        "runtime_seconds": round(elapsed, 2),
        "images_per_second": round(len(todo) / elapsed, 3) if todo and elapsed else None,
        "prompts_over_max_sequence_length": n_long,
        "store": stat,
    })
    previews = export_pair_previews(
        store, datasets, config.output_dir / "images_preview", config.images.preview_pairs, config.runtime.seed
    )
    manifest.payload["previews"] = [str(p) for p in previews]
    manifest.finish()
    store.close()
    return {
        "store": str(store.directory),
        "n_unique_texts": len(texts),
        "n_generated": len(todo),
        "n_reused": len(texts) - len(todo),
        "runtime_seconds": round(elapsed, 2),
        "store_images": stat["n_images"],
        "store_shards": stat["n_shards"],
        "store_gb": round(stat["bytes"] / 1024 ** 3, 3),
        "previews": str(config.output_dir / "images_preview") if previews else None,
        "manifest": str(manifest.path),
    }


# --------------------------------------------------------------------------- #
# Preview sheets
# --------------------------------------------------------------------------- #
def _pair_sheet(store: ImageStore, pair: STSPair):
    """The pair's two images side by side, its sentences and gold score below."""
    from PIL import Image, ImageDraw

    left = store.load(pair.sentence1, _PREVIEW_SIDE)
    right = store.load(pair.sentence2, _PREVIEW_SIDE)
    gap, margin = 8, 10
    width = left.width + right.width + gap
    font = font_of_size(16)
    line_height = 21
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    header = f"{pair.pair_id}  |  {pair.domain or pair.dataset}  |  gold {pair.gold_score:g}"
    lines = [header]
    for label, sentence in (("1", pair.sentence1), ("2", pair.sentence2)):
        lines += wrap_text(measure, f"{label}: {sentence}", font, width - 2 * margin)

    top = max(left.height, right.height)
    canvas = Image.new("RGB", (width, top + 2 * margin + line_height * len(lines)), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    draw = ImageDraw.Draw(canvas)
    for row, line in enumerate(lines):
        draw.text((margin, top + margin + row * line_height), line, fill="black", font=font)
    return canvas


def export_pair_previews(
    store: ImageStore,
    datasets: Dict[str, STSDataset],
    directory: Path,
    per_dataset: int,
    seed: int,
) -> List[Path]:
    """Write ``per_dataset`` pair sheets per dataset plus a README index."""
    if per_dataset <= 0:
        return []
    directory.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    rows: List[str] = []
    for name, dataset in datasets.items():
        # subsample() stratifies over the domain, so CxC shows both sampling methods.
        for pair in dataset.subsample(per_dataset, seed=seed).pairs:
            if pair.sentence1 not in store or pair.sentence2 not in store:
                continue
            path = directory / f"{pair.pair_id}.jpg"
            _pair_sheet(store, pair).save(path, quality=90)
            written.append(path)
            cells = [f"[{pair.pair_id}]({path.name})", pair.domain or "—", f"{pair.gold_score:g}",
                     pair.sentence1, pair.sentence2]
            rows.append("| " + " | ".join(c.replace("|", "\\|") for c in cells) + " |")
    if written:
        (directory / "README.md").write_text("\n".join([
            "# Generated-image previews",
            "",
            f"Generator store: `{store.directory}`",
            "",
            "| pair | domain | gold | sentence 1 | sentence 2 |",
            "|---|---|---|---|---|",
            *rows,
            "",
        ]), encoding="utf-8")
        _log.info("Image previews -> %s (%d pairs)", directory, len(written))
    return written
