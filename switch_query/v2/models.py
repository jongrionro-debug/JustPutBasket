"""Data models and interfaces for the V2 text/tag retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, Sequence

from switch_query.tagging.attributes import DEFAULT_ATTRIBUTE_NAMES

ATTRIBUTE_NAMES = tuple(DEFAULT_ATTRIBUTE_NAMES)
MULTI_VALUE_ATTRIBUTES = {"category", "silhouette", "mood", "detail"}
MULTI_VALUE_SEPARATOR = "|"
Stage = Literal["mood_board", "sketch_stage"]


@dataclass(slots=True)
class V2PipelineInput:
    query_text: str
    stage: Stage
    balance_score: float
    user_uploaded_image: str | None = None


@dataclass(slots=True)
class V2ParsedQuery:
    query_text: str
    canonical_tags: dict[str, str] = field(default_factory=dict)
    raw_phrases: dict[str, str] = field(default_factory=dict)
    required_features: list[str] = field(default_factory=list)
    preferred_features: list[str] = field(default_factory=list)
    confidence: float = 0.0
    query_document: str = ""


@dataclass(slots=True)
class V2ArchiveDocument:
    image_id: str
    file_path: str
    brand: str
    season_group: str
    canonical_tags: dict[str, str] = field(default_factory=dict)
    raw_tags: dict[str, str] = field(default_factory=dict)
    document_text: str = ""


@dataclass(slots=True)
class V2IndexedDocument(V2ArchiveDocument):
    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class V2ArchiveIndex:
    documents: list[V2IndexedDocument] = field(default_factory=list)
    feature_vocabulary: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class V2RankedResult:
    image_id: str
    file_path: str
    brand: str
    season_group: str
    score: float
    matched_attributes: dict[str, str] = field(default_factory=dict)
    mismatched_attributes: dict[str, str] = field(default_factory=dict)
    missing_attributes: dict[str, str] = field(default_factory=dict)
    score_breakdown: dict[str, float] = field(default_factory=dict)
    match_reasons: list[str] = field(default_factory=list)
    explanation: str = ""


@dataclass(slots=True)
class V2PipelineOutput:
    parsed_query: V2ParsedQuery
    top_results: list[V2RankedResult]
    retrieval_metadata: dict[str, object] = field(default_factory=dict)


class TextEncoder(Protocol):
    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        """Return dense vectors for text inputs."""


class QueryParser(Protocol):
    def parse(
        self,
        query_text: str,
        *,
        stage: str,
        balance_score: float,
        user_uploaded_image: str | None = None,
    ) -> V2ParsedQuery:
        """Return a structured query representation."""


class LocalIndexStore(Protocol):
    def save(self, index: V2ArchiveIndex) -> None:
        """Persist the V2 archive index."""

    def load(self) -> V2ArchiveIndex:
        """Load the V2 archive index."""
