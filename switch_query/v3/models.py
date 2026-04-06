"""Data models for the V3 item-aware retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

ExtractionMode = Literal["text_only", "image_assisted"]
V3ItemMatchStatus = Literal[
    "exact",
    "partial",
    "missing",
    "contradiction",
    "fallback_match",
    "hard_fail",
]


@dataclass(slots=True)
class V3TargetItem:
    """Structured query target item for item-aware matching."""

    target_item_id: str
    category: str
    color: list[str] = field(default_factory=list)
    silhouette: list[str] = field(default_factory=list)
    material: list[str] = field(default_factory=list)
    pattern: list[str] = field(default_factory=list)
    texture: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    required_attributes: list[str] = field(default_factory=list)
    preferred_attributes: list[str] = field(default_factory=list)
    raw_phrase: str = ""


@dataclass(slots=True)
class V3ParsedQuery:
    """Structured V3 query with item-level intent binding."""

    query_text: str
    target_items: list[V3TargetItem] = field(default_factory=list)
    global_constraints: dict[str, list[str]] = field(default_factory=dict)
    style_preferences: dict[str, list[str]] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass(slots=True)
class V3ItemExtractionInput:
    """Extractor input derived from existing normalized archive metadata."""

    image_id: str
    file_path: str
    brand: str
    season_group: str
    canonical_tags: dict[str, str] = field(default_factory=dict)
    raw_tags: dict[str, str] = field(default_factory=dict)
    detail: str = ""
    image_path: str | None = None
    extraction_mode: ExtractionMode = "text_only"


@dataclass(slots=True)
class V3DocumentItem:
    """Structured item extracted from a single archive document."""

    item_id: str
    category: str
    color: list[str] = field(default_factory=list)
    silhouette: list[str] = field(default_factory=list)
    material: list[str] = field(default_factory=list)
    pattern: list[str] = field(default_factory=list)
    texture: list[str] = field(default_factory=list)
    style_tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    source: str = ""


@dataclass(slots=True)
class V3ItemExtractionOutput:
    """Extractor output that enriches a document with item-level structure."""

    items: list[V3DocumentItem] = field(default_factory=list)
    item_confidence: float = 0.0
    item_extraction_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V3ArchiveDocument:
    """Archive document enriched with item-level structure."""

    image_id: str
    file_path: str
    brand: str
    season_group: str
    canonical_tags: dict[str, str] = field(default_factory=dict)
    raw_tags: dict[str, str] = field(default_factory=dict)
    detail: str = ""
    items: list[V3DocumentItem] = field(default_factory=list)
    item_confidence: float = 0.0
    item_extraction_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V3IndexedDocument(V3ArchiveDocument):
    """Archive document with optional dense vector features."""

    vector: list[float] = field(default_factory=list)


@dataclass(slots=True)
class V3ArchiveIndex:
    """Persisted V3 archive index payload."""

    documents: list[V3IndexedDocument] = field(default_factory=list)
    feature_vocabulary: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(slots=True)
class V3ItemAssignment:
    """Per-target item match result for one ranked document."""

    target_item_id: str
    target_category: str
    status: V3ItemMatchStatus
    matched_item_id: str = ""
    matched_category: str = ""
    source: str = "none"
    score: float = 0.0
    matched_attributes: dict[str, str] = field(default_factory=dict)
    missing_attributes: list[str] = field(default_factory=list)
    contradicted_attributes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class V3RankedResult:
    """Ranked symbolic result for the V3 item-aware retriever."""

    image_id: str
    file_path: str
    brand: str
    season_group: str
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    match_reasons: list[str] = field(default_factory=list)
    explanation: str = ""
    item_assignments: list[V3ItemAssignment] = field(default_factory=list)
