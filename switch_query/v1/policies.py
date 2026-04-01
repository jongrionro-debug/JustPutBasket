"""Policy defaults for the V1 retrieval scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from .models import RetrievalMode

DEFAULT_TEXT_WEIGHT = 0.35
DEFAULT_IMAGE_WEIGHT = 0.65
DEFAULT_TOP_K = 20


@dataclass(slots=True)
class V1PipelineConfig:
    top_k: int = DEFAULT_TOP_K
    retrieval_mode: RetrievalMode = "fusion"
    text_weight: float = DEFAULT_TEXT_WEIGHT
    image_weight: float = DEFAULT_IMAGE_WEIGHT


def balance_bucket(balance_score: float) -> str:
    if balance_score <= -0.35:
        return "divergent"
    if balance_score >= 0.35:
        return "convergent"
    return "neutral"


def synthetic_reference_count(balance_score: float) -> int:
    if balance_score <= -0.70:
        return 4
    if balance_score <= -0.15:
        return 3
    return 1


def average_dense_vectors(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = len(vectors[0])
    totals = [0.0] * width
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("All dense vectors must have the same dimensionality")
        for index, value in enumerate(vector):
            totals[index] += value
    return [value / len(vectors) for value in totals]
