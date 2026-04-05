from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from switch_query.tagging.preprocessing import NormalizedTagRow
from switch_query.v2 import (
    InMemoryArchiveIndex,
    JsonArchiveIndexStore,
    V2Pipeline,
    V2PipelineConfig,
    V2PipelineInput,
    V2QueryParser,
    build_archive_documents,
    build_archive_index,
    build_feature_vocabulary,
    dense_cosine_similarity,
    load_archive_documents,
    write_archive_documents,
)
from switch_query.v2.explanation import explain_match


class FakeTextEncoder:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


def build_rows(tmp_path: Path) -> list[NormalizedTagRow]:
    return [
        NormalizedTagRow(
            image_id="look-1",
            file_path=str(tmp_path / "look-1.jpg"),
            season_group="spring-ready-to-wear",
            year="2026",
            brand="alpha",
            source_type="collection",
            filename="look-1.jpg",
            caption="black wool coat with sharp tailoring",
            raw_category="coat",
            raw_silhouette="sharp tailoring",
            raw_color="jet black",
            raw_material="wool",
            raw_pattern="solid",
            raw_texture="smooth",
            raw_mood="minimal but sharp",
            raw_season="fall",
            raw_era="modern",
            raw_detail="long coat|wide leg trousers",
            review_needed="false",
            confidence_note="high",
            canonical_category="coat",
            canonical_silhouette="tailored",
            canonical_color="black",
            canonical_material="wool",
            canonical_pattern="solid",
            canonical_texture="smooth",
            canonical_mood="minimal|sharp",
            canonical_season="fall",
            canonical_era="modern",
            canonical_detail="long coat|wide leg trousers",
        ),
        NormalizedTagRow(
            image_id="look-2",
            file_path=str(tmp_path / "look-2.jpg"),
            season_group="spring-ready-to-wear",
            year="2026",
            brand="beta",
            source_type="collection",
            filename="look-2.jpg",
            caption="red silk dress with romantic mood",
            raw_category="gown",
            raw_silhouette="fluid",
            raw_color="scarlet",
            raw_material="silk",
            raw_pattern="solid",
            raw_texture="satin",
            raw_mood="romantic",
            raw_season="spring",
            raw_era="vintage",
            raw_detail="evening gown|heels",
            review_needed="false",
            confidence_note="high",
            canonical_category="dress",
            canonical_silhouette="fluid",
            canonical_color="red",
            canonical_material="silk",
            canonical_pattern="solid",
            canonical_texture="satin",
            canonical_mood="romantic",
            canonical_season="spring",
            canonical_era="vintage",
            canonical_detail="gown|heels",
        ),
    ]


def test_build_archive_documents_preserves_raw_and_canonical_fields(tmp_path: Path) -> None:
    rows = build_rows(tmp_path)

    documents = build_archive_documents(rows)

    assert len(documents) == 2
    assert documents[0].canonical_tags["mood"] == "minimal|sharp"
    assert documents[0].raw_tags["mood"] == "minimal but sharp"
    lines = documents[0].document_text.splitlines()
    assert lines[:4] == [
        "category: coat",
        "silhouette: tailored",
        "color: black",
        "material: wool",
    ]
    assert "caption: black wool coat with sharp tailoring" in lines
    assert "raw_mood: minimal but sharp" in lines
    assert lines[-2:] == ["brand: alpha", "season_group: spring-ready-to-wear"]


def test_archive_document_json_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    documents = build_archive_documents(build_rows(tmp_path))

    write_archive_documents(str(path), documents)
    restored = load_archive_documents(str(path))

    assert restored[0].image_id == "look-1"
    assert restored[1].canonical_tags["category"] == "dress"


def test_build_feature_vocabulary_includes_raw_variants(tmp_path: Path) -> None:
    rows = build_rows(tmp_path)

    vocabulary = build_feature_vocabulary(rows=rows)

    assert vocabulary["color"]["jet black"] == "black"
    assert vocabulary["mood"]["minimal but sharp"] == "minimal|sharp"
    assert vocabulary["category"]["gown"] == "dress"


def test_query_parser_extracts_multi_value_tags_and_preserves_raw_phrase(tmp_path: Path) -> None:
    rows = build_rows(tmp_path)
    parser = V2QueryParser(build_feature_vocabulary(rows=rows))

    parsed = parser.parse(
        "Black tailored coat with minimal but sharp mood",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.canonical_tags["color"] == "black"
    assert parsed.canonical_tags["silhouette"] == "tailored"
    assert parsed.canonical_tags["mood"] == "minimal|sharp"
    assert parsed.raw_phrases["mood"] == "minimal but sharp"
    assert "query_text: Black tailored coat with minimal but sharp mood" in parsed.query_document


def test_query_parser_leaves_unknown_fields_empty(tmp_path: Path) -> None:
    rows = build_rows(tmp_path)
    parser = V2QueryParser(build_feature_vocabulary(rows=rows))

    parsed = parser.parse(
        "architectural editorial energy",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.canonical_tags == {}
    assert parsed.raw_phrases == {}


def test_explain_match_splits_matched_mismatched_and_missing() -> None:
    matched, mismatched, missing, explanation = explain_match(
        {"color": "black", "mood": "minimal|sharp", "detail": "wide leg trousers"},
        {"color": "black", "mood": "minimal|romantic"},
    )

    assert matched == {"color": "black", "mood": "minimal"}
    assert mismatched == {"mood": "minimal|romantic"}
    assert missing == {"detail": "wide leg trousers"}
    assert "matched: color=black, mood=minimal" in explanation
    assert "mismatched: mood=minimal|romantic" in explanation
    assert "missing: detail=wide leg trousers" in explanation


def test_dense_cosine_similarity_matches_expected_value() -> None:
    score = dense_cosine_similarity([1.0, 0.0], [1.0, 1.0])

    assert score == pytest.approx(0.70710678)


def test_build_archive_index_writes_json_index(tmp_path: Path) -> None:
    documents = build_archive_documents(build_rows(tmp_path))
    index_path = tmp_path / "archive_index.json"
    encoder = FakeTextEncoder(
        {
            documents[0].document_text: [1.0, 0.0],
            documents[1].document_text: [0.0, 1.0],
        }
    )

    build_archive_index(documents, encoder, JsonArchiveIndexStore(str(index_path)))

    payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert payload["documents"][0]["image_id"] == "look-1"
    assert payload["documents"][1]["vector"] == [0.0, 1.0]
    assert payload["feature_vocabulary"]["color"]["jet black"] == "black"


def test_pipeline_runs_end_to_end_and_returns_explanations(tmp_path: Path) -> None:
    documents = build_archive_documents(build_rows(tmp_path))
    query_text = "Black tailored coat with minimal but sharp mood"
    query_document = (
        "category: coat\n"
        "silhouette: tailored\n"
        "color: black\n"
        "mood: minimal|sharp\n"
        "raw_mood: minimal but sharp\n"
        "raw_silhouette: tailored\n"
        f"query_text: {query_text}"
    )
    encoder = FakeTextEncoder(
        {
            documents[0].document_text: [1.0, 0.0],
            documents[1].document_text: [0.0, 1.0],
            query_document: [1.0, 0.0],
        }
    )
    store = InMemoryArchiveIndex()
    build_archive_index(documents, encoder, store)
    pipeline = V2Pipeline(
        encoder=encoder,
        index_store=store,
        config=V2PipelineConfig(top_k=2),
    )

    output = pipeline.run(
        V2PipelineInput(
            query_text=query_text,
            stage="mood_board",
            balance_score=0.3,
            user_uploaded_image=str(tmp_path / "uploaded.jpg"),
        )
    )

    assert output.parsed_query.canonical_tags["color"] == "black"
    assert output.top_results[0].image_id == "look-1"
    assert output.top_results[0].score == pytest.approx(1.0)
    assert output.top_results[0].matched_attributes["mood"] == "minimal|sharp"
    assert output.top_results[0].explanation.startswith("matched:")
    assert output.retrieval_metadata["uploaded_image_used_in_scoring"] is False
    assert output.retrieval_metadata["used_uploaded_image"] is True
    assert [result.image_id for result in output.top_results] == ["look-1", "look-2"]


def test_pipeline_limits_results_to_top_k(tmp_path: Path) -> None:
    rows = build_rows(tmp_path)
    extra_row = NormalizedTagRow(
        **{
            **asdict(rows[1]),
            "image_id": "look-3",
            "file_path": str(tmp_path / "look-3.jpg"),
            "filename": "look-3.jpg",
            "brand": "gamma",
        }
    )
    documents = build_archive_documents([rows[0], rows[1], extra_row])
    query_text = "Black tailored coat with minimal but sharp mood"
    query_document = (
        "category: coat\n"
        "silhouette: tailored\n"
        "color: black\n"
        "mood: minimal|sharp\n"
        "raw_mood: minimal but sharp\n"
        "raw_silhouette: tailored\n"
        f"query_text: {query_text}"
    )
    encoder = FakeTextEncoder(
        {
            documents[0].document_text: [1.0, 0.0],
            documents[1].document_text: [0.0, 1.0],
            documents[2].document_text: [0.2, 0.8],
            query_document: [1.0, 0.0],
        }
    )
    store = InMemoryArchiveIndex()
    build_archive_index(documents, encoder, store)
    pipeline = V2Pipeline(
        encoder=encoder,
        index_store=store,
        config=V2PipelineConfig(top_k=2),
    )

    output = pipeline.run(
        V2PipelineInput(
            query_text=query_text,
            stage="mood_board",
            balance_score=0.0,
        )
    )

    assert len(output.top_results) == 2
    assert [result.image_id for result in output.top_results] == ["look-1", "look-3"]
