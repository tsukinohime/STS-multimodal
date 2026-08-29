"""Model adapters: one frozen pretrained encoder behind a common interface."""

from .base import EncodeOutput, TextEncoder, build_encoder, l2_normalize

__all__ = ["EncodeOutput", "TextEncoder", "build_encoder", "l2_normalize"]
