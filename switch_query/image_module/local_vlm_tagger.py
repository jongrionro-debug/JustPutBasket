"""Compatibility wrapper for the local VLM tagger."""

from switch_query.tagging.local_vlm_tagger import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    _coerce_json,
    main,
    run_local_vlm,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_PROMPT",
    "_coerce_json",
    "main",
    "run_local_vlm",
]
