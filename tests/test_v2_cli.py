from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from switch_query.tagging import NormalizedTagRow, write_csv
from switch_query.v2 import cli
from switch_query.v2.models import V2ParsedQuery
import switch_query.v2.pipeline as v2_pipeline


class FakeBuildEncoder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device or "cpu"

    def encode_text(self, texts):
        return [[float(index), 0.0] for index, _text in enumerate(texts, start=1)]


class FakeQueryEncoder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device or "cpu"

    def encode_text(self, texts):
        mapping = {
            "category: coat\nsilhouette: tailored\ncolor: black\nmood: minimal|sharp\nraw_mood: minimal but sharp\nraw_silhouette: tailored\nquery_text: Black tailored coat with minimal but sharp mood": [1.0, 0.0],
        }
        vectors = []
        for text in texts:
            vectors.append(mapping.get(text, [1.0, 0.0] if "category: coat" in text else [0.0, 1.0]))
        return vectors


class StubParser:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def parse(self, query_text, *, stage, balance_score, user_uploaded_image=None):
        del stage, balance_score, user_uploaded_image
        return V2ParsedQuery(
            query_text=query_text,
            canonical_tags={
                "category": "coat",
                "silhouette": "tailored",
                "color": "black",
                "mood": "minimal|sharp",
            },
            raw_phrases={
                "mood": "minimal but sharp",
                "silhouette": "tailored",
            },
            required_features=["category", "color"],
            preferred_features=["silhouette", "mood"],
            confidence=0.92,
            query_document=(
                "category: coat\n"
                "silhouette: tailored\n"
                "color: black\n"
                "mood: minimal|sharp\n"
                "raw_mood: minimal but sharp\n"
                "raw_silhouette: tailored\n"
                f"query_text: {query_text}"
            ),
        )


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


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_build_archive_index_writes_json_index_and_documents(tmp_path: Path, monkeypatch) -> None:
    normalized_tags_path = tmp_path / "normalized.csv"
    index_path = tmp_path / "archive_index.json"
    documents_path = tmp_path / "documents.json"
    write_csv(normalized_tags_path, build_rows(tmp_path))

    monkeypatch.setattr(cli, "SigLIP2TextEncoder", FakeBuildEncoder)

    result = cli.build_archive_index(
        normalized_tags_path=str(normalized_tags_path),
        output_path=str(index_path),
        documents_output_path=str(documents_path),
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        use_embeddings=True,
    )

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    document_payload = json.loads(documents_path.read_text(encoding="utf-8"))

    assert result.document_count == 2
    assert result.brand_count == 2
    assert result.embeddings_enabled is True
    assert result.model_name == "fake/siglip2"
    assert result.device == "cpu"
    assert index_payload["documents"][0]["vector"] == [1.0, 0.0]
    assert index_payload["feature_vocabulary"]["color"]["jet black"] == "black"
    assert document_payload[0]["canonical_tags"]["mood"] == "minimal|sharp"


def test_build_archive_index_supports_vectorless_mode(tmp_path: Path) -> None:
    normalized_tags_path = tmp_path / "normalized.csv"
    index_path = tmp_path / "archive_index.json"
    write_csv(normalized_tags_path, build_rows(tmp_path))

    result = cli.build_archive_index(
        normalized_tags_path=str(normalized_tags_path),
        output_path=str(index_path),
        documents_output_path=None,
        model_name=None,
        device=None,
        batch_size=4,
        use_embeddings=False,
    )

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))

    assert result.embeddings_enabled is False
    assert result.model_name == ""
    assert result.device == ""
    assert index_payload["documents"][0]["vector"] == []


def test_build_archive_index_rejects_empty_normalized_csv(tmp_path: Path, monkeypatch) -> None:
    normalized_tags_path = tmp_path / "normalized.csv"
    normalized_tags_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "SigLIP2TextEncoder", FakeBuildEncoder)

    try:
        cli.build_archive_index(
            normalized_tags_path=str(normalized_tags_path),
            output_path=str(tmp_path / "archive_index.json"),
            documents_output_path=None,
            model_name="fake/siglip2",
            device="cpu",
            batch_size=4,
            use_embeddings=True,
        )
    except ValueError as exc:
        assert "No normalized rows found" in str(exc)
    else:
        raise AssertionError("Expected build_archive_index to reject an empty normalized tag file")


def test_run_query_writes_ranked_csv(tmp_path: Path, monkeypatch) -> None:
    create_image(tmp_path / "look-1.jpg")
    create_image(tmp_path / "look-2.jpg")
    index_path = tmp_path / "archive_index.json"
    index_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "image_id": "look-1",
                        "file_path": str(tmp_path / "look-1.jpg"),
                        "brand": "alpha",
                        "season_group": "spring-ready-to-wear",
                        "canonical_tags": {
                            "category": "coat",
                            "silhouette": "tailored",
                            "color": "black",
                            "mood": "minimal|sharp",
                        },
                        "raw_tags": {
                            "mood": "minimal but sharp",
                            "silhouette": "sharp tailoring",
                        },
                        "document_text": "category: coat\nsilhouette: tailored\ncolor: black\nmood: minimal|sharp\nraw_mood: minimal but sharp\nbrand: alpha\nseason_group: spring-ready-to-wear",
                        "vector": [1.0, 0.0],
                    },
                    {
                        "image_id": "look-2",
                        "file_path": str(tmp_path / "look-2.jpg"),
                        "brand": "beta",
                        "season_group": "spring-ready-to-wear",
                        "canonical_tags": {"category": "dress", "color": "red"},
                        "raw_tags": {"mood": "romantic"},
                        "document_text": "category: dress\ncolor: red\nbrand: beta\nseason_group: spring-ready-to-wear",
                        "vector": [0.0, 1.0],
                    },
                ],
                "feature_vocabulary": {
                    "category": {"coat": "coat", "gown": "dress", "dress": "dress"},
                    "silhouette": {"tailored": "tailored", "sharp tailoring": "tailored"},
                    "color": {"black": "black", "jet black": "black", "red": "red"},
                    "material": {},
                    "pattern": {},
                    "texture": {},
                    "mood": {"minimal but sharp": "minimal|sharp", "romantic": "romantic"},
                    "season": {},
                    "era": {},
                    "detail": {},
                },
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "results.csv"

    monkeypatch.setattr(cli, "LuxiaQueryParser", StubParser)

    result = cli.run_query(
        index_path=str(index_path),
        query_text="Black tailored coat with minimal but sharp mood",
        stage="mood_board",
        balance_score=0.0,
        user_uploaded_image=None,
        output_path=str(output_path),
        html_output_path=None,
        parser_provider="luxia",
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=2,
    )

    rows = output_path.read_text(encoding="utf-8").splitlines()

    assert result.result_count == 1
    assert result.parser_provider == "luxia"
    assert rows[0] == "rank,image_id,score,brand,season_group,file_path,matched_attributes,mismatched_attributes,missing_attributes,score_breakdown,match_reasons,explanation"
    assert "look-1" in rows[1]
    assert "color=black" in rows[1]
    assert "mood=minimal|sharp" in rows[1]
    assert "category:exact=+8.0" in rows[1]
    assert "category exact match" in rows[1]
    assert all("look-2" not in row for row in rows[1:])


def test_run_query_writes_html_preview(tmp_path: Path, monkeypatch) -> None:
    create_image(tmp_path / "look-1.jpg")
    create_image(tmp_path / "look-2.jpg")
    index_path = tmp_path / "archive_index.json"
    row = build_rows(tmp_path)[0]
    index_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "image_id": row.image_id,
                        "file_path": row.file_path,
                        "brand": row.brand,
                        "season_group": row.season_group,
                        "canonical_tags": {
                            "category": row.canonical_category,
                            "silhouette": row.canonical_silhouette,
                            "color": row.canonical_color,
                            "mood": row.canonical_mood,
                        },
                        "raw_tags": {"mood": row.raw_mood, "silhouette": row.raw_silhouette},
                        "document_text": "category: coat\nsilhouette: tailored\ncolor: black\nmood: minimal|sharp\nraw_mood: minimal but sharp\nbrand: alpha\nseason_group: spring-ready-to-wear",
                        "vector": [1.0, 0.0],
                    }
                ],
                "feature_vocabulary": {
                    "category": {"coat": "coat"},
                    "silhouette": {"tailored": "tailored"},
                    "color": {"black": "black"},
                    "material": {},
                    "pattern": {},
                    "texture": {},
                    "mood": {"minimal but sharp": "minimal|sharp"},
                    "season": {},
                    "era": {},
                    "detail": {},
                },
            }
        ),
        encoding="utf-8",
    )
    html_output_path = tmp_path / "results.html"
    monkeypatch.setattr(cli, "LuxiaQueryParser", StubParser)

    result = cli.run_query(
        index_path=str(index_path),
        query_text="Black tailored coat with minimal but sharp mood",
        stage="mood_board",
        balance_score=0.0,
        user_uploaded_image=str(tmp_path / "uploaded.jpg"),
        output_path=None,
        html_output_path=str(html_output_path),
        parser_provider="luxia",
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=1,
    )

    html = html_output_path.read_text(encoding="utf-8")

    assert result.html_output_path == str(html_output_path.resolve())
    assert "V2 Retrieval Preview" in html
    assert "Black tailored coat with minimal but sharp mood" in html
    assert "matched_required: category=coat, color=black" in html
    assert "score_breakdown" in html
    assert "match_reasons" in html
    assert Path(build_rows(tmp_path)[0].file_path).resolve().as_uri() in html


def test_run_query_set_writes_manifest_and_candidate_judgments(tmp_path: Path, monkeypatch) -> None:
    create_image(tmp_path / "look-1.jpg")
    create_image(tmp_path / "look-2.jpg")
    index_path = tmp_path / "archive_index.json"
    index_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "image_id": "look-1",
                        "file_path": str(tmp_path / "look-1.jpg"),
                        "brand": "alpha",
                        "season_group": "spring-ready-to-wear",
                        "canonical_tags": {
                            "category": "coat",
                            "silhouette": "tailored",
                            "color": "black",
                            "mood": "minimal|sharp",
                        },
                        "raw_tags": {
                            "mood": "minimal but sharp",
                            "silhouette": "sharp tailoring",
                        },
                        "document_text": "category: coat\nsilhouette: tailored\ncolor: black",
                        "vector": [1.0, 0.0],
                    },
                    {
                        "image_id": "look-2",
                        "file_path": str(tmp_path / "look-2.jpg"),
                        "brand": "beta",
                        "season_group": "spring-ready-to-wear",
                        "canonical_tags": {"category": "dress", "color": "red"},
                        "raw_tags": {"mood": "romantic"},
                        "document_text": "category: dress\ncolor: red",
                        "vector": [0.0, 1.0],
                    },
                ],
                "feature_vocabulary": {
                    "category": {"coat": "coat", "dress": "dress"},
                    "silhouette": {"tailored": "tailored"},
                    "color": {"black": "black", "red": "red"},
                    "material": {},
                    "pattern": {},
                    "texture": {},
                    "mood": {"minimal but sharp": "minimal|sharp", "romantic": "romantic"},
                    "season": {},
                    "era": {},
                    "detail": {},
                },
            }
        ),
        encoding="utf-8",
    )
    queries_path = tmp_path / "queries.csv"
    queries_path.write_text(
        "\n".join(
            [
                "query_id,query_text,stage,query_type,expected_failure_type",
                "q001,Black tailored coat with minimal but sharp mood,mood_board,color_sensitive,item_color_ambiguity",
                "q002,Another black tailored coat,mood_board,mixed,outfit_level_color_mismatch",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "eval"

    monkeypatch.setattr(cli, "LuxiaQueryParser", StubParser)

    result = cli.run_query_set(
        index_path=str(index_path),
        queries_path=str(queries_path),
        output_dir=str(output_dir),
        parser_provider="luxia",
        model_name=None,
        device=None,
        batch_size=4,
        top_k=3,
    )

    manifest_path = output_dir / "query_manifest.csv"
    candidate_judgments_path = output_dir / "candidate_judgments.csv"
    results_dir = output_dir / "results"

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with open(candidate_judgments_path, newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.DictReader(handle))

    assert result.query_count == 2
    assert result.candidate_count == 2
    assert manifest_rows[0]["query_id"] == "q001"
    assert manifest_rows[0]["csv_output_path"].endswith("q001.csv")
    assert manifest_rows[0]["html_output_path"].endswith("q001.html")
    assert candidate_rows[0]["query_id"] == "q001"
    assert candidate_rows[0]["image_id"] == "look-1"
    assert candidate_rows[0]["label"] == ""
    assert candidate_rows[0]["failure_type"] == ""
    assert candidate_rows[0]["notes"] == ""
    assert (results_dir / "q001.csv").exists()
    assert (results_dir / "q001.html").exists()
