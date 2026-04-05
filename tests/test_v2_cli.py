from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from switch_query.tagging import NormalizedTagRow, write_csv
from switch_query.v2 import cli


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
    )

    index_payload = json.loads(index_path.read_text(encoding="utf-8"))
    document_payload = json.loads(documents_path.read_text(encoding="utf-8"))

    assert result.document_count == 2
    assert result.brand_count == 2
    assert result.model_name == "fake/siglip2"
    assert result.device == "cpu"
    assert index_payload["documents"][0]["vector"] == [1.0, 0.0]
    assert index_payload["feature_vocabulary"]["color"]["jet black"] == "black"
    assert document_payload[0]["canonical_tags"]["mood"] == "minimal|sharp"


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

    monkeypatch.setattr(cli, "SigLIP2TextEncoder", FakeQueryEncoder)

    result = cli.run_query(
        index_path=str(index_path),
        query_text="Black tailored coat with minimal but sharp mood",
        stage="mood_board",
        balance_score=0.0,
        user_uploaded_image=None,
        output_path=str(output_path),
        html_output_path=None,
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=2,
    )

    rows = output_path.read_text(encoding="utf-8").splitlines()

    assert result.result_count == 2
    assert rows[0] == "rank,image_id,score,brand,season_group,file_path,matched_attributes,mismatched_attributes,missing_attributes,explanation"
    assert "look-1" in rows[1]
    assert "color=black" in rows[1]
    assert "mood=minimal|sharp" in rows[1]


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
    monkeypatch.setattr(cli, "SigLIP2TextEncoder", FakeQueryEncoder)

    result = cli.run_query(
        index_path=str(index_path),
        query_text="Black tailored coat with minimal but sharp mood",
        stage="mood_board",
        balance_score=0.0,
        user_uploaded_image=str(tmp_path / "uploaded.jpg"),
        output_path=None,
        html_output_path=str(html_output_path),
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=1,
    )

    html = html_output_path.read_text(encoding="utf-8")

    assert result.html_output_path == str(html_output_path.resolve())
    assert "V2 Retrieval Preview" in html
    assert "Black tailored coat with minimal but sharp mood" in html
    assert "matched: category=coat" in html
    assert Path(build_rows(tmp_path)[0].file_path).resolve().as_uri() in html
