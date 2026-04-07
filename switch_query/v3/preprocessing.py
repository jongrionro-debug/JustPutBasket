"""Helpers for preparing V3 item extraction inputs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from switch_query.tagging.preprocessing import NormalizedTagRow

from .concepts import (
    COLOR_TOKENS,
    extract_color_style_tags,
    extract_style_concepts,
    is_apparel_category,
    normalize_token,
    split_multi_value_text,
)
from .item_extractor import extraction_input_from_normalized_row
from .models import (
    ExtractionMode,
    V3ArchiveDocument,
    V3DocumentItem,
    V3ItemExtractionInput,
    V3ItemExtractionOutput,
)

DEFAULT_OUTPUT_ROOT = "tmp/v3_preprocessing"
ATTRIBUTE_BACKFILL_NOTE = "item attributes backfilled from canonical/raw mood-era-detail context"


@dataclass(slots=True)
class V3PreprocessingPaths:
    normalized_tags_path: str
    root_dir: str
    item_inputs_sample_path: str
    item_inputs_full_path: str
    item_outputs_sample_path: str
    item_outputs_full_path: str
    item_enriched_documents_sample_path: str
    item_enriched_documents_full_path: str


def build_preprocessing_paths(
    normalized_tags_path: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
) -> V3PreprocessingPaths:
    normalized_path = Path(normalized_tags_path).resolve()
    dataset_slug = normalized_path.parent.name
    root_dir = Path(output_root).resolve() / dataset_slug
    return V3PreprocessingPaths(
        normalized_tags_path=str(normalized_path),
        root_dir=str(root_dir),
        item_inputs_sample_path=str(root_dir / "item_inputs_sample.jsonl"),
        item_inputs_full_path=str(root_dir / "item_inputs_full.jsonl"),
        item_outputs_sample_path=str(root_dir / "item_outputs_sample.jsonl"),
        item_outputs_full_path=str(root_dir / "item_outputs_full.jsonl"),
        item_enriched_documents_sample_path=str(root_dir / "item_enriched_documents_sample.jsonl"),
        item_enriched_documents_full_path=str(root_dir / "item_enriched_documents_full.jsonl"),
    )


def read_normalized_tag_rows(path: str) -> list[NormalizedTagRow]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [NormalizedTagRow(**row) for row in reader]


def build_item_extraction_inputs(
    rows: list[NormalizedTagRow],
    *,
    extraction_mode: ExtractionMode = "text_only",
    offset: int = 0,
    limit: int | None = None,
) -> list[V3ItemExtractionInput]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    selected_rows = rows[offset:]
    if limit is not None:
        selected_rows = selected_rows[:limit]
    return [
        extraction_input_from_normalized_row(row, extraction_mode=extraction_mode)
        for row in selected_rows
    ]


def write_item_extraction_inputs_jsonl(
    path: str,
    inputs: list[V3ItemExtractionInput],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for item in inputs:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def read_item_extraction_inputs_jsonl(path: str) -> list[V3ItemExtractionInput]:
    with open(path, encoding="utf-8") as handle:
        return [
            V3ItemExtractionInput(**json.loads(line))
            for line in handle
            if line.strip()
        ]


def slice_item_extraction_inputs(
    inputs: list[V3ItemExtractionInput],
    *,
    offset: int = 0,
    limit: int | None = None,
) -> list[V3ItemExtractionInput]:
    if offset < 0:
        raise ValueError("offset must be >= 0")
    selected_inputs = inputs[offset:]
    if limit is not None:
        selected_inputs = selected_inputs[:limit]
    return selected_inputs


def write_item_extraction_outputs_jsonl(
    path: str,
    outputs: list[V3ItemExtractionOutput],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for item in outputs:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def append_item_extraction_output_jsonl(
    path: str,
    output: V3ItemExtractionOutput,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(output), ensure_ascii=False) + "\n")


def read_item_extraction_outputs_jsonl(path: str) -> list[V3ItemExtractionOutput]:
    with open(path, encoding="utf-8") as handle:
        return [
            _coerce_item_extraction_output(json.loads(line))
            for line in handle
            if line.strip()
        ]


def merge_item_inputs_and_outputs(
    inputs: list[V3ItemExtractionInput],
    outputs: list[V3ItemExtractionOutput],
) -> list[V3ArchiveDocument]:
    if len(inputs) != len(outputs):
        raise ValueError(
            "Item extraction inputs and outputs must have the same length for merge."
        )

    documents: list[V3ArchiveDocument] = []
    for extraction_input, extraction_output in zip(inputs, outputs, strict=True):
        documents.append(
            V3ArchiveDocument(
                image_id=extraction_input.image_id,
                file_path=extraction_input.file_path,
                brand=extraction_input.brand,
                season_group=extraction_input.season_group,
                canonical_tags=dict(extraction_input.canonical_tags),
                raw_tags=dict(extraction_input.raw_tags),
                detail=extraction_input.detail,
                items=list(extraction_output.items),
                item_confidence=extraction_output.item_confidence,
                item_extraction_notes=list(extraction_output.item_extraction_notes),
            )
        )
    return documents


def write_archive_documents_jsonl(
    path: str,
    documents: list[V3ArchiveDocument],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


def read_archive_documents_jsonl(path: str) -> list[V3ArchiveDocument]:
    with open(path, encoding="utf-8") as handle:
        return [
            _coerce_archive_document(json.loads(line))
            for line in handle
            if line.strip()
        ]


def backfill_archive_document_style_tags(
    documents: list[V3ArchiveDocument],
) -> list[V3ArchiveDocument]:
    updated_documents: list[V3ArchiveDocument] = []
    for document in documents:
        inferred_style_tags = _infer_document_style_tags(document)
        inferred_style_concepts = _infer_document_style_concepts(document)
        if not document.items:
            updated_documents.append(document)
            continue

        apparel_item_count = sum(1 for item in document.items if is_apparel_category(item.category))
        canonical_colors = split_multi_value_text(document.canonical_tags.get("color", ""))
        updated_items: list[V3DocumentItem] = []
        changed = False
        for item in document.items:
            merged_style_tags = list(item.style_tags)
            for style_tag in inferred_style_tags:
                if style_tag not in merged_style_tags:
                    merged_style_tags.append(style_tag)

            merged_style_concepts = list(item.style_concepts)
            if is_apparel_category(item.category):
                for style_concept in inferred_style_concepts:
                    if style_concept not in merged_style_concepts:
                        merged_style_concepts.append(style_concept)

            merged_colors = list(item.color)
            if not merged_colors:
                merged_colors = _infer_item_backfill_colors(
                    document,
                    item,
                    canonical_colors=canonical_colors,
                    apparel_item_count=apparel_item_count,
                )

            if (
                merged_style_tags != item.style_tags
                or merged_style_concepts != item.style_concepts
                or merged_colors != item.color
            ):
                changed = True
                updated_items.append(
                    replace(
                        item,
                        color=merged_colors,
                        style_tags=merged_style_tags,
                        style_concepts=merged_style_concepts,
                    )
                )
            else:
                updated_items.append(item)

        if not changed:
            updated_documents.append(document)
            continue

        notes = list(document.item_extraction_notes)
        if ATTRIBUTE_BACKFILL_NOTE not in notes:
            notes.append(ATTRIBUTE_BACKFILL_NOTE)
        updated_documents.append(
            replace(
                document,
                items=updated_items,
                item_extraction_notes=notes,
            )
        )
    return updated_documents


def _coerce_item_extraction_output(payload: dict[str, object]) -> V3ItemExtractionOutput:
    raw_items = payload.get("items", [])
    items = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if isinstance(raw_item, dict):
                from .models import V3DocumentItem

                items.append(V3DocumentItem(**raw_item))
    return V3ItemExtractionOutput(
        items=items,
        item_confidence=float(payload.get("item_confidence", 0.0) or 0.0),
        item_extraction_notes=[
            str(note)
            for note in payload.get("item_extraction_notes", [])
            if str(note).strip()
        ]
        if isinstance(payload.get("item_extraction_notes", []), list)
        else [],
    )


def _coerce_archive_document(payload: dict[str, object]) -> V3ArchiveDocument:
    raw_items = payload.get("items", [])
    items = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if isinstance(raw_item, dict):
                items.append(V3DocumentItem(**raw_item))
    return V3ArchiveDocument(
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
    )


def _infer_document_style_tags(document: V3ArchiveDocument) -> list[str]:
    context_parts = [
        document.detail,
        document.canonical_tags.get("detail", ""),
        document.raw_tags.get("detail", ""),
    ]
    context = " ".join(part.strip() for part in context_parts if part and part.strip())
    return extract_color_style_tags(context)


def _infer_document_style_concepts(document: V3ArchiveDocument) -> list[str]:
    context_parts = [
        document.detail,
        document.canonical_tags.get("mood", ""),
        document.canonical_tags.get("era", ""),
        document.canonical_tags.get("detail", ""),
        document.raw_tags.get("mood", ""),
        document.raw_tags.get("era", ""),
        document.raw_tags.get("detail", ""),
    ]
    context = " ".join(part.strip() for part in context_parts if part and part.strip())
    return extract_style_concepts(context)


def _infer_item_backfill_colors(
    document: V3ArchiveDocument,
    item: V3DocumentItem,
    *,
    canonical_colors: list[str],
    apparel_item_count: int,
) -> list[str]:
    detail_colors = _infer_colors_from_detail_phrase(document.detail, item.category)
    if detail_colors:
        return detail_colors[:2]
    if is_apparel_category(item.category) and apparel_item_count == 1 and len(canonical_colors) == 1:
        return canonical_colors[:1]
    return []


def _infer_colors_from_detail_phrase(detail: str, category: str) -> list[str]:
    normalized_category = normalize_token(category)
    category_tokens = {
        normalized_category,
        normalized_category.rstrip("s"),
        f"{normalized_category}s",
    }
    for phrase in split_multi_value_text(detail):
        normalized_phrase = normalize_token(phrase)
        if not any(token and token in normalized_phrase for token in category_tokens):
            continue
        colors = [color for color in COLOR_TOKENS if f" {color} " in f" {normalized_phrase} "]
        if not colors:
            continue
        deduped: list[str] = []
        for color in colors:
            if color not in deduped:
                deduped.append(color)
        return deduped[:2]
    return []


def _normalize(value: str) -> str:
    return normalize_token(value)
