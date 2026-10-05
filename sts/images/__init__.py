"""Generated images: one per sentence, made once and shared by every image condition.

``store``       content-addressed, sharded-tar image store (+ resizing helpers)
``generators``  text-to-image models (FLUX.2 [klein]) and a model-free stand-in
``pipeline``    the ``imagine`` stage: generate what is missing, write previews

``pipeline`` is not imported here: it depends on :mod:`sts.pipeline`, which
itself reads the store.
"""

from .generators import SEED_POLICIES, ImageGenerator, build_generator, seed_for
from .store import ImageStore, MissingImagesError, fit_image, iter_prefetched, open_image_store, visual_tokens

__all__ = [
    "SEED_POLICIES",
    "ImageGenerator",
    "build_generator",
    "seed_for",
    "ImageStore",
    "MissingImagesError",
    "fit_image",
    "iter_prefetched",
    "open_image_store",
    "visual_tokens",
]
