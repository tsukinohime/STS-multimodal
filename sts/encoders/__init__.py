"""Text encoders behind a common :class:`TextEncoder` interface."""

from .base import TextEncoder, build_encoder

__all__ = ["TextEncoder", "build_encoder"]
