"""Storage adapters for V3 archive indexes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import V3ArchiveIndex, V3DocumentItem, V3IndexedDocument


@dataclass(slots=True)
class InMemoryArchiveIndex:
    index: V3ArchiveIndex = field(default_factory=V3ArchiveIndex)

    def save(self, index: V3ArchiveIndex) -> None:
        self.index = index

    def load(self) -> V3ArchiveIndex:
        return V3ArchiveIndex(
            documents=[_clone_indexed_document(document) for document in self.index.documents],
            feature_vocabulary={
                feature: dict(values) for feature, values in self.index.feature_vocabulary.items()
            },
        )


@dataclass(slots=True)
class JsonArchiveIndexStore:
    path: str

    def save(self, index: V3ArchiveIndex) -> None:
        destination = Path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "documents": [asdict(document) for document in index.documents],
                    "feature_vocabulary": index.feature_vocabulary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> V3ArchiveIndex:
        source = Path(self.path)
        if not source.exists():
            return V3ArchiveIndex()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return V3ArchiveIndex(
            documents=[
                _coerce_indexed_document(document)
                for document in payload.get("documents", [])
                if isinstance(document, dict)
            ],
            feature_vocabulary=payload.get("feature_vocabulary", {}),
        )


def _clone_indexed_document(document: V3IndexedDocument) -> V3IndexedDocument:
    return V3IndexedDocument(
        image_id=document.image_id,
        file_path=document.file_path,
        brand=document.brand,
        season_group=document.season_group,
        canonical_tags=dict(document.canonical_tags),
        raw_tags=dict(document.raw_tags),
        detail=document.detail,
        items=[
            V3DocumentItem(
                item_id=item.item_id,
                category=item.category,
                color=list(item.color),
                silhouette=list(item.silhouette),
                material=list(item.material),
                pattern=list(item.pattern),
                texture=list(item.texture),
                style_tags=list(item.style_tags),
                confidence=item.confidence,
                evidence=list(item.evidence),
                source=item.source,
            )
            for item in document.items
        ],
        item_confidence=document.item_confidence,
        item_extraction_notes=list(document.item_extraction_notes),
        vector=list(document.vector),
    )


def _coerce_indexed_document(payload: dict[str, object]) -> V3IndexedDocument:
    raw_items = payload.get("items", [])
    items = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if isinstance(raw_item, dict):
                items.append(V3DocumentItem(**raw_item))
    return V3IndexedDocument(
        image_id=str(payload.get("image_id", "")),
        file_path=str(payload.get("file_path", "")),
        brand=str(payload.get("brand", "")),
        season_group=str(payload.get("season_group", "")),
        canonical_tags=dict(payload.get("canonical_tags", {}))
        if isinstance(payload.get("canonical_tags", {}), dict)
        else {},
        raw_tags=dict(payload.get("raw_tags", {}))
        if isinstance(payload.get("raw_tags", {}), dict)
        else {},
        detail=str(payload.get("detail", "")),
        items=items,
        item_confidence=float(payload.get("item_confidence", 0.0) or 0.0),
        item_extraction_notes=[
            str(note)
            for note in payload.get("item_extraction_notes", [])
            if str(note).strip()
        ]
        if isinstance(payload.get("item_extraction_notes", []), list)
        else [],
        vector=[
            float(value)
            for value in payload.get("vector", [])
            if isinstance(value, (int, float))
        ]
        if isinstance(payload.get("vector", []), list)
        else [],
    )
