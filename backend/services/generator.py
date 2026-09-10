"""Backward-compatible import path for the enhanced story generation engine."""

from backend.services.generator_v2 import generate_content, normalize_category

__all__ = ["generate_content", "normalize_category"]
