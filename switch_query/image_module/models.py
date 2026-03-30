"""Data models for the image module pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Stage = Literal["mood_board", "sketch_stage"]
FeedbackEventType = Literal["select", "save", "exclude"]
GeneratedRole = Literal["synthetic_reference", "inspiration"]


@dataclass(slots=True)
class ImageRef:
    image_id: str
    description: str = ""
    attributes: dict[str, str] = field(default_factory=dict)
    source: str = "user_upload"
    local_path: str | None = None
    image_url: str | None = None


@dataclass(slots=True)
class ImageRecord:
    image_id: str
    source: str
    attributes: dict[str, str]
    embedding: dict[str, float]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RankedImage:
    image_id: str
    source: str
    final_score: float
    embedding_score: float
    tag_score: float
    matched_attributes: dict[str, str]
    penalized_attributes: dict[str, str]
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class GeneratedImage:
    generated_id: str
    role: GeneratedRole
    prompt_summary: str
    balance_bucket: str
    used_for_retrieval: bool
    attributes: dict[str, str]
    image_base64: str | None = None


@dataclass(slots=True)
class ImageModuleInput:
    query_text: str
    user_uploaded_images: list[ImageRef]
    stage: Stage
    balance_score: float


@dataclass(slots=True)
class ImageModuleOutput:
    archive_results: list[RankedImage]
    generated_results: list[GeneratedImage]


@dataclass(slots=True)
class FeedbackEvent:
    event_type: FeedbackEventType
    target_id: str
    target_kind: Literal["archive", "generated"]
    query_text: str
    stage: Stage
    balance_score: float
