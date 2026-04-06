from __future__ import annotations

import json
from pathlib import Path

from switch_query.tagging import NormalizedTagRow, write_csv
from switch_query.v3.preprocessing import (
    build_item_extraction_inputs,
    build_preprocessing_paths,
    merge_item_inputs_and_outputs,
    slice_item_extraction_inputs,
    read_normalized_tag_rows,
    write_archive_documents_jsonl,
    write_item_extraction_inputs_jsonl,
    write_item_extraction_outputs_jsonl,
)
from switch_query.v3.models import V3DocumentItem, V3ItemExtractionOutput


def test_v3_preprocessing_builds_item_input_jsonl(tmp_path: Path) -> None:
    normalized_path = tmp_path / "normalized_tags.csv"
    write_csv(
        normalized_path,
        [
            NormalizedTagRow(
                image_id="2026:spring-ready-to-wear:test-brand:0001",
                file_path=str(tmp_path / "look-1.jpg"),
                season_group="spring-ready-to-wear",
                year="2026",
                brand="test-brand",
                source_type="collection",
                filename="look-1.jpg",
                caption="black jacket with wide-leg white trousers",
                raw_category="jacket|trousers",
                raw_silhouette="wide-leg",
                raw_color="black|white",
                raw_material="",
                raw_pattern="",
                raw_texture="",
                raw_mood="minimal",
                raw_season="spring",
                raw_era="modern",
                raw_detail="black jacket|wide-leg white trousers",
                review_needed="false",
                confidence_note="high",
                canonical_category="jacket|trousers",
                canonical_silhouette="wide-leg",
                canonical_color="black|white",
                canonical_material="",
                canonical_pattern="",
                canonical_texture="",
                canonical_mood="minimal",
                canonical_season="spring",
                canonical_era="modern",
                canonical_detail="black jacket|wide-leg white trousers",
            )
        ],
    )

    paths = build_preprocessing_paths(str(normalized_path), output_root=str(tmp_path / "out"))
    rows = read_normalized_tag_rows(str(normalized_path))
    inputs = build_item_extraction_inputs(rows, extraction_mode="text_only")
    write_item_extraction_inputs_jsonl(paths.item_inputs_sample_path, inputs)

    payload_lines = Path(paths.item_inputs_sample_path).read_text(encoding="utf-8").splitlines()
    payload = json.loads(payload_lines[0])

    assert len(inputs) == 1
    assert payload["image_id"] == "2026:spring-ready-to-wear:test-brand:0001"
    assert payload["canonical_tags"]["category"] == "jacket|trousers"
    assert payload["detail"] == "black jacket|wide-leg white trousers"
    assert payload["extraction_mode"] == "text_only"


def test_v3_preprocessing_merges_inputs_and_outputs_into_archive_documents(tmp_path: Path) -> None:
    normalized_path = tmp_path / "normalized_tags.csv"
    write_csv(
        normalized_path,
        [
            NormalizedTagRow(
                image_id="2026:spring-ready-to-wear:test-brand:0002",
                file_path=str(tmp_path / "look-2.jpg"),
                season_group="spring-ready-to-wear",
                year="2026",
                brand="test-brand",
                source_type="collection",
                filename="look-2.jpg",
                caption="minimal black coat",
                raw_category="coat",
                raw_silhouette="tailored",
                raw_color="black",
                raw_material="wool",
                raw_pattern="solid",
                raw_texture="smooth",
                raw_mood="minimal",
                raw_season="spring",
                raw_era="modern",
                raw_detail="minimal black coat",
                review_needed="false",
                confidence_note="high",
                canonical_category="coat",
                canonical_silhouette="tailored",
                canonical_color="black",
                canonical_material="wool",
                canonical_pattern="solid",
                canonical_texture="smooth",
                canonical_mood="minimal",
                canonical_season="spring",
                canonical_era="modern",
                canonical_detail="minimal black coat",
            )
        ],
    )

    paths = build_preprocessing_paths(str(normalized_path), output_root=str(tmp_path / "out"))
    rows = read_normalized_tag_rows(str(normalized_path))
    inputs = build_item_extraction_inputs(rows, extraction_mode="image_assisted")
    outputs = [
        V3ItemExtractionOutput(
            items=[
                V3DocumentItem(
                    item_id="2026:spring-ready-to-wear:test-brand:0002#1",
                    category="coat",
                    color=["black"],
                    silhouette=["tailored"],
                    style_tags=["minimal"],
                    confidence=0.93,
                    evidence=["detail:minimal black coat"],
                    source="luxia_image_assisted",
                )
            ],
            item_confidence=0.93,
            item_extraction_notes=["clean single-item extraction"],
        )
    ]

    write_item_extraction_inputs_jsonl(paths.item_inputs_sample_path, inputs)
    write_item_extraction_outputs_jsonl(paths.item_outputs_sample_path, outputs)
    documents = merge_item_inputs_and_outputs(inputs, outputs)
    write_archive_documents_jsonl(paths.item_enriched_documents_sample_path, documents)

    payload_lines = Path(paths.item_enriched_documents_sample_path).read_text(encoding="utf-8").splitlines()
    payload = json.loads(payload_lines[0])

    assert len(documents) == 1
    assert payload["image_id"] == "2026:spring-ready-to-wear:test-brand:0002"
    assert payload["items"][0]["category"] == "coat"
    assert payload["items"][0]["item_id"].endswith("#1")
    assert payload["item_confidence"] == 0.93


def test_v3_preprocessing_supports_offset_slicing_for_chunk_runs(tmp_path: Path) -> None:
    rows = [
        NormalizedTagRow(
            image_id=f"2026:spring-ready-to-wear:test-brand:{index:04d}",
            file_path=str(tmp_path / f"look-{index}.jpg"),
            season_group="spring-ready-to-wear",
            year="2026",
            brand="test-brand",
            source_type="collection",
            filename=f"look-{index}.jpg",
            caption="sample",
            raw_category="coat",
            raw_silhouette="tailored",
            raw_color="black",
            raw_material="",
            raw_pattern="",
            raw_texture="",
            raw_mood="minimal",
            raw_season="spring",
            raw_era="modern",
            raw_detail="black coat",
            review_needed="false",
            confidence_note="high",
            canonical_category="coat",
            canonical_silhouette="tailored",
            canonical_color="black",
            canonical_material="",
            canonical_pattern="",
            canonical_texture="",
            canonical_mood="minimal",
            canonical_season="spring",
            canonical_era="modern",
            canonical_detail="black coat",
        )
        for index in range(3)
    ]

    inputs = build_item_extraction_inputs(rows, extraction_mode="text_only")
    sliced_inputs = slice_item_extraction_inputs(inputs, offset=1, limit=1)

    assert len(inputs) == 3
    assert len(sliced_inputs) == 1
    assert sliced_inputs[0].image_id.endswith(":0001")
