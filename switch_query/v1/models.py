"""Models and interfaces for the V1 retrieval scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

RetrievalMode = Literal["text_only", "image_only", "fusion"]


@dataclass(slots=True)
class V1ArchiveRecord:
    image_id: str
    file_path: str
    brand: str
    vector: list[float]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedReference:
    generated_id: str
    prompt_summary: str
    image_path: str
    balance_bucket: str


@dataclass(slots=True)
class V1QueryArtifacts:
    query_text: str
    text_vector: list[float]
    generated_image_vectors: list[list[float]]
    merged_generated_image_vector: list[float]


@dataclass(slots=True)
class V1ScoredCandidate:
    image_id: str
    file_path: str
    brand: str
    final_score: float
    text_score: float
    image_score: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class V1RankedResult(V1ScoredCandidate):
    query_id: str = ""
    query_text: str = ""
    rank: int = 0


@dataclass(slots=True)
class V1PipelineInput:
    query_id: str
    query_text: str
    balance_score: float
    generated_image_paths: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V1PipelineOutput:
    retrieval_mode: RetrievalMode
    generated_references: list[GeneratedReference]
    query_artifacts: V1QueryArtifacts
    archive_results: list[V1RankedResult]


class MultimodalEncoder(Protocol):
    """Text and image encoders must come from the same model family."""

    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Return dense vectors for query texts."""

    def encode_image(self, image_paths: Sequence[str]) -> list[list[float]]:
        """Return dense vectors for archive or generated images."""


class SyntheticImageGenerator(Protocol):
    def generate(self, query_text: str, count: int) -> list[str]:
        """Return paths or opaque handles for generated reference images."""


class LocalIndexStore(Protocol):
    def save(self, records: list[V1ArchiveRecord]) -> None:
        """Persist archive vectors plus metadata."""

    def load(self) -> list[V1ArchiveRecord]:
        """Load archive vectors plus metadata."""
