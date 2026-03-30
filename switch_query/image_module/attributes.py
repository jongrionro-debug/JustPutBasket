"""Canonical attributes and stage-aware weighting."""

from __future__ import annotations

from typing import Mapping

DEFAULT_ATTRIBUTE_NAMES = (
    "category",
    "silhouette",
    "color",
    "material",
    "pattern",
    "texture",
    "mood",
    "season",
    "era",
    "detail",
)

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


def balance_bucket(balance_score: float) -> str:
    """Bucket the balance signal for generated output metadata."""
    if balance_score <= -0.35:
        return "divergent"
    if balance_score >= 0.35:
        return "convergent"
    return "neutral"


def synthetic_reference_count(balance_score: float) -> int:
    """Return the number of synthetic references for a balance score."""
    if balance_score <= -0.70:
        return 4
    if balance_score <= -0.15:
        return 3
    return 1


def stage_attribute_weight(stage: str, attribute_name: str) -> float:
    return DEFAULT_STAGE_ATTRIBUTE_WEIGHTS.get(stage, {}).get(attribute_name, 1.0)


def stage_similarity_weight(stage: str) -> float:
    return 1.10 if stage == "sketch_stage" else 1.0


def attribute_importance_map(stage: str) -> Mapping[str, float]:
    weights = {attribute: 1.0 for attribute in DEFAULT_ATTRIBUTE_NAMES}
    weights.update(DEFAULT_STAGE_ATTRIBUTE_WEIGHTS.get(stage, {}))
    return weights
