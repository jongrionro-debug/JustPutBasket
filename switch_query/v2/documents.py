"""Offline document builders for the V2 text/tag retrieval pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

from switch_query.tagging.preprocessing import CanonicalMappingRow, NormalizedTagRow

from .models import ATTRIBUTE_NAMES, MULTI_VALUE_SEPARATOR, V2ArchiveDocument


def build_archive_documents(rows: Sequence[NormalizedTagRow]) -> list[V2ArchiveDocument]:
    documents: list[V2ArchiveDocument] = []
    for row in rows:
        canonical_tags = {
            feature: getattr(row, f"canonical_{feature}", "").strip()
            for feature in ATTRIBUTE_NAMES
            if getattr(row, f"canonical_{feature}", "").strip()
        }
        raw_tags = {
            feature: getattr(row, f"raw_{feature}", "").strip()
            for feature in ATTRIBUTE_NAMES
            if getattr(row, f"raw_{feature}", "").strip()
        }
        documents.append(
            V2ArchiveDocument(
                image_id=row.image_id,
                file_path=row.file_path,
                brand=row.brand,
                season_group=row.season_group,
                canonical_tags=canonical_tags,
                raw_tags=raw_tags,
                document_text=compose_archive_document_text(
                    canonical_tags=canonical_tags,
                    raw_tags=raw_tags,
                    brand=row.brand,
                    season_group=row.season_group,
                    caption=row.caption,
                ),
            )
        )
    return documents


def compose_archive_document_text(
    *,
    canonical_tags: dict[str, str],
    raw_tags: dict[str, str],
    brand: str,
    season_group: str,
    caption: str = "",
) -> str:
    lines: list[str] = []
    for feature in ATTRIBUTE_NAMES:
        value = canonical_tags.get(feature, "").strip()
        if value:
            lines.append(f"{feature}: {value}")

    if caption.strip():
        lines.append(f"caption: {caption.strip()}")

    for feature in ("mood", "detail"):
        raw_value = raw_tags.get(feature, "").strip()
        if raw_value:
            lines.append(f"raw_{feature}: {raw_value}")

    if brand.strip():
        lines.append(f"brand: {brand.strip()}")
    if season_group.strip():
        lines.append(f"season_group: {season_group.strip()}")
    return "\n".join(lines)


def compose_query_document_text(
    *,
    query_text: str,
    canonical_tags: dict[str, str],
    raw_phrases: dict[str, str],
) -> str:
    lines: list[str] = []
    for feature in ATTRIBUTE_NAMES:
        value = canonical_tags.get(feature, "").strip()
        if value:
            lines.append(f"{feature}: {value}")

    for feature in ("mood", "detail", "silhouette"):
        raw_value = raw_phrases.get(feature, "").strip()
        if raw_value:
            lines.append(f"raw_{feature}: {raw_value}")

    lines.append(f"query_text: {query_text.strip()}")
    return "\n".join(line for line in lines if line)


def build_feature_vocabulary(
    rows: Sequence[NormalizedTagRow] | None = None,
    mappings: Sequence[CanonicalMappingRow] | None = None,
    documents: Sequence[V2ArchiveDocument] | None = None,
) -> dict[str, dict[str, str]]:
    vocabulary = {feature: {} for feature in ATTRIBUTE_NAMES}

    if rows is not None:
        for row in rows:
            for feature in ATTRIBUTE_NAMES:
                canonical_values = _split_values(getattr(row, f"canonical_{feature}", ""))
                raw_values = _split_values(getattr(row, f"raw_{feature}", ""))
                for value in canonical_values:
                    vocabulary[feature][_normalize_token(value)] = value
                if len(canonical_values) == len(raw_values):
                    for raw_value, canonical_value in zip(raw_values, canonical_values, strict=True):
                        vocabulary[feature][_normalize_token(raw_value)] = canonical_value
                elif len(raw_values) == 1 and canonical_values:
                    vocabulary[feature][_normalize_token(raw_values[0])] = MULTI_VALUE_SEPARATOR.join(
                        canonical_values
                    )
                else:
                    for raw_value in raw_values:
                        vocabulary[feature].setdefault(_normalize_token(raw_value), raw_value)

    if documents is not None:
        for document in documents:
            for feature in ATTRIBUTE_NAMES:
                for value in _split_values(document.canonical_tags.get(feature, "")):
                    vocabulary[feature][_normalize_token(value)] = value
                raw_values = _split_values(document.raw_tags.get(feature, ""))
                canonical_values = _split_values(document.canonical_tags.get(feature, ""))
                if len(canonical_values) == len(raw_values):
                    for raw_value, canonical_value in zip(raw_values, canonical_values, strict=True):
                        vocabulary[feature][_normalize_token(raw_value)] = canonical_value
                elif len(raw_values) == 1 and canonical_values:
                    vocabulary[feature][_normalize_token(raw_values[0])] = MULTI_VALUE_SEPARATOR.join(
                        canonical_values
                    )

    if mappings is not None:
        for mapping in mappings:
            if mapping.status.lower() == "rejected":
                continue
            feature = mapping.feature
            if feature not in vocabulary:
                continue
            vocabulary[feature][_normalize_token(mapping.canonical)] = mapping.canonical
            vocabulary[feature][_normalize_token(mapping.variant)] = mapping.canonical

    return vocabulary


def write_archive_documents(path: str, documents: Sequence[V2ArchiveDocument]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([asdict(document) for document in documents], indent=2),
        encoding="utf-8",
    )


def load_archive_documents(path: str) -> list[V2ArchiveDocument]:
    source = Path(path)
    if not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    return [V2ArchiveDocument(**row) for row in payload]


def _split_values(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(MULTI_VALUE_SEPARATOR) if item.strip()]


def _normalize_token(value: str) -> str:
    return " ".join(value.lower().strip().replace("-", " ").split())
