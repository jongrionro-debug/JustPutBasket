"""Compatibility helpers for the legacy image_module namespace."""

from __future__ import annotations

from typing import Mapping

from switch_query.tagging.attributes import DEFAULT_ATTRIBUTE_NAMES
from switch_query.v1.policies import balance_bucket, synthetic_reference_count

DEFAULT_STAGE_ATTRIBUTE_WEIGHTS: dict[str, dict[str, float]] = {
    "mood_board": {
        "mood": 1.45,
        "color": 1.25,
        "season": 1.15,
        "pattern": 1.10,
    },
    "sketch_stage": {
        "silhouette": 1.50,
        "material": 1.30,
        "detail": 1.25,
        "texture": 1.10,
    },
}


def stage_attribute_weight(stage: str, attribute_name: str) -> float:
    return DEFAULT_STAGE_ATTRIBUTE_WEIGHTS.get(stage, {}).get(attribute_name, 1.0)


def stage_similarity_weight(stage: str) -> float:
    return 1.10 if stage == "sketch_stage" else 1.0


def attribute_importance_map(stage: str) -> Mapping[str, float]:
    weights = {attribute: 1.0 for attribute in DEFAULT_ATTRIBUTE_NAMES}
    weights.update(DEFAULT_STAGE_ATTRIBUTE_WEIGHTS.get(stage, {}))
    return weights
