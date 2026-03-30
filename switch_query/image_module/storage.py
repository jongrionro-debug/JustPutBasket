"""Storage primitives that mirror relational and vector responsibilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt

from .models import FeedbackEvent, ImageRecord
from .synonyms import SynonymCatalog


@dataclass(slots=True)
class InMemoryRelationalStore:
    images: dict[str, ImageRecord] = field(default_factory=dict)

    def add_image(self, image: ImageRecord) -> None:
        self.images[image.image_id] = image

    def get_image(self, image_id: str) -> ImageRecord:
        return self.images[image_id]

    def list_images(self) -> list[ImageRecord]:
        return list(self.images.values())

    def update_attributes(self, image_id: str, attributes: dict[str, str]) -> None:
        image = self.images[image_id]
        image.attributes = dict(attributes)


@dataclass(slots=True)
class InMemoryVectorStore:
    vectors: dict[str, dict[str, float]] = field(default_factory=dict)

    def add_vector(self, image_id: str, vector: dict[str, float]) -> None:
        self.vectors[image_id] = vector

    def search(self, query_vector: dict[str, float], limit: int) -> list[tuple[str, float]]:
        scored = [
            (image_id, cosine_similarity(query_vector, vector))
            for image_id, vector in self.vectors.items()
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


@dataclass(slots=True)
class InMemoryFeedbackStore:
    events: list[FeedbackEvent] = field(default_factory=list)

    def record(self, event: FeedbackEvent) -> None:
        self.events.append(event)


def cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    overlap = set(left).intersection(right)
    numerator = sum(left[key] * right[key] for key in overlap)
    left_norm = sqrt(sum(value * value for value in left.values()))
    right_norm = sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_archive_image(
    image_id: str,
    attributes: dict[str, str],
    synonym_catalog: SynonymCatalog,
    source: str = "vogue_runway",
    metadata: dict[str, str] | None = None,
) -> ImageRecord:
    normalized_attributes = synonym_catalog.normalize_attributes(attributes)
    embedding = {
        f"{name}:{value}": 1.0
        for name, value in normalized_attributes.items()
        if value
    }
    return ImageRecord(
        image_id=image_id,
        source=source,
        attributes=normalized_attributes,
        embedding=embedding,
        metadata=metadata or {},
    )
