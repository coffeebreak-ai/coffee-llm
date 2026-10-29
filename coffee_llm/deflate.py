"""Compatibility exports for the dialogue deflation implementation."""

from coffee_llm.dialogue.deflate import (
    DEFLATE_ADJASTOR,
    DEFLATOR,
    _assistant_text,
    deflate_text,
)

__all__ = ["DEFLATE_ADJASTOR", "DEFLATOR", "_assistant_text", "deflate_text"]
