"""Index builders for the V3 item-aware retrieval pipeline."""

from __future__ import annotations

from typing import Sequence

from switch_query.tagging.attributes import DEFAULT_ATTRIBUTE_NAMES

from .models import MultimodalEncoder, V3ArchiveDocument, V3ArchiveIndex, V3IndexedDocument

ATTRIBUTE_NAMES = tuple(DEFAULT_ATTRIBUTE_NAMES)
MULTI_VALUE_SEPARATOR = "|"


def build_archive_index(
    documents: Sequence[V3ArchiveDocument],
    encoder: MultimodalEncoder,
    index_store,
) -> V3ArchiveIndex:
    image_paths = [document.file_path for document in documents]
    vectors = encoder.encode_image(image_paths)
    if len(vectors) != len(documents):
        raise RuntimeError("Image encoder output size did not match archive document count")

    indexed_documents = [
        V3IndexedDocument(
            image_id=document.image_id,
            file_path=document.file_path,
            brand=document.brand,
            season_group=document.season_group,
            canonical_tags=dict(document.canonical_tags),
            raw_tags=dict(document.raw_tags),
            detail=document.detail,
            items=list(document.items),
            item_confidence=document.item_confidence,
            item_extraction_notes=list(document.item_extraction_notes),
            vector=vector,
        )
        for document, vector in zip(documents, vectors, strict=True)
    ]
    index = V3ArchiveIndex(
        documents=indexed_documents,
        feature_vocabulary=build_feature_vocabulary(documents),
    )
    index_store.save(index)
    return index


def build_feature_vocabulary(
    documents: Sequence[V3ArchiveDocument],
) -> dict[str, dict[str, str]]:
    vocabulary = {feature: {} for feature in ATTRIBUTE_NAMES}

    for document in documents:
        for feature in ATTRIBUTE_NAMES:
            canonical_values = _split_values(document.canonical_tags.get(feature, ""))
            raw_values = _split_values(document.raw_tags.get(feature, ""))
            for value in canonical_values:
                vocabulary[feature][_normalize_token(value)] = value
            if len(canonical_values) == len(raw_values):
                for raw_value, canonical_value in zip(raw_values, canonical_values, strict=True):
                    vocabulary[feature][_normalize_token(raw_value)] = canonical_value
            elif len(raw_values) == 1 and canonical_values:
                vocabulary[feature][_normalize_token(raw_values[0])] = MULTI_VALUE_SEPARATOR.join(
                    canonical_values
                )

    return vocabulary


def _split_values(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(MULTI_VALUE_SEPARATOR) if item.strip()]


def _normalize_token(value: str) -> str:
    return " ".join(value.lower().strip().replace("-", " ").split())
