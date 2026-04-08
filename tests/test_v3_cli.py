from __future__ import annotations

import csv
import json
from pathlib import Path

from switch_query.v3 import cli
from switch_query.v3.models import V3ParsedQuery, V3TargetItem


class FakeBuildEncoder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device or "cpu"

    def encode_image(self, image_paths):
        return [[float(index), 0.0] for index, _path in enumerate(image_paths, start=1)]

    def encode_text(self, texts):
        return [[1.0, 0.0] for _text in texts]


class FakeQueryEncoder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device or "cpu"

    def encode_image(self, image_paths):
        return [[0.0, 0.0] for _path in image_paths]

    def encode_text(self, texts):
        assert texts == ["item1 ; trousers ; color black"]
        return [[1.0, 0.0]]


class StubParser:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def parse(self, query_text, *, stage, balance_score, user_uploaded_image=None):
        del stage, balance_score, user_uploaded_image
        return V3ParsedQuery(
            query_text=query_text,
            target_items=[
                V3TargetItem(
                    target_item_id="item_1",
                    category="trousers",
                    color=["black"],
                    required_attributes=["category", "color"],
                    raw_phrase="black trousers",
                )
            ],
            global_constraints={},
            style_preferences={},
            confidence=0.9,
        )


def build_documents_jsonl(tmp_path: Path) -> Path:
    documents_path = tmp_path / "item_enriched_documents.jsonl"
    documents_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "image_id": "look-1",
                        "file_path": str(tmp_path / "look-1.jpg"),
                        "brand": "alpha",
                        "season_group": "spring-ready-to-wear",
                        "canonical_tags": {"category": "trousers", "color": "black"},
                        "raw_tags": {},
                        "detail": "black trousers",
                        "items": [],
                        "item_confidence": 0.0,
                        "item_extraction_notes": [],
                    }
                ),
                json.dumps(
                    {
                        "image_id": "look-2",
                        "file_path": str(tmp_path / "look-2.jpg"),
                        "brand": "beta",
                        "season_group": "spring-ready-to-wear",
                        "canonical_tags": {"category": "trousers", "color": "white"},
                        "raw_tags": {},
                        "detail": "white trousers",
                        "items": [],
                        "item_confidence": 0.0,
                        "item_extraction_notes": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return documents_path


def build_index_json(tmp_path: Path) -> Path:
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
                        "canonical_tags": {"category": "trousers", "color": "black"},
                        "raw_tags": {},
                        "detail": "black trousers",
                        "items": [],
                        "item_confidence": 0.0,
                        "item_extraction_notes": [],
                        "vector": [1.0, 0.0],
                    },
                    {
                        "image_id": "look-2",
                        "file_path": str(tmp_path / "look-2.jpg"),
                        "brand": "beta",
                        "season_group": "spring-ready-to-wear",
                        "canonical_tags": {"category": "trousers", "color": "white"},
                        "raw_tags": {},
                        "detail": "white trousers",
                        "items": [],
                        "item_confidence": 0.0,
                        "item_extraction_notes": [],
                        "vector": [0.0, 1.0],
                    },
                ],
                "feature_vocabulary": {
                    "category": {"trousers": "trousers"},
                    "silhouette": {},
                    "color": {"black": "black", "white": "white"},
                    "material": {},
                    "pattern": {},
                    "texture": {},
                    "mood": {},
                    "season": {},
                    "era": {},
                    "detail": {},
                },
            }
        ),
        encoding="utf-8",
    )
    return index_path


def test_v3_cli_build_archive_index_writes_json_index(tmp_path: Path, monkeypatch) -> None:
    documents_path = build_documents_jsonl(tmp_path)
    output_path = tmp_path / "archive_index.json"
    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeBuildEncoder)

    result = cli.build_archive_index(
        documents_path=str(documents_path),
        output_path=str(output_path),
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.document_count == 2
    assert result.brand_count == 2
    assert payload["documents"][0]["vector"] == [1.0, 0.0]
    assert payload["feature_vocabulary"]["color"]["black"] == "black"


def test_v3_cli_backfill_style_tags_writes_updated_documents_jsonl(tmp_path: Path) -> None:
    documents_path = tmp_path / "item_enriched_documents.jsonl"
    documents_path.write_text(
        json.dumps(
            {
                "image_id": "look-1",
                "file_path": str(tmp_path / "look-1.jpg"),
                "brand": "alpha",
                "season_group": "spring-ready-to-wear",
                "canonical_tags": {"era": "vintage"},
                "raw_tags": {"mood": "worn-in vintage"},
                "detail": "vintage washed pants",
                "items": [
                    {
                        "item_id": "look-1#1",
                        "category": "pants",
                        "color": [],
                        "silhouette": [],
                        "material": [],
                        "pattern": [],
                        "texture": [],
                        "style_tags": [],
                        "confidence": 0.9,
                        "evidence": [],
                        "source": "test",
                    }
                ],
                "item_confidence": 0.9,
                "item_extraction_notes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "item_enriched_documents_backfilled.jsonl"

    result = cli.backfill_style_tags(
        input_path=str(documents_path),
        output_path=str(output_path),
    )

    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

    assert result.document_count == 1
    assert result.updated_document_count == 1
    assert result.updated_item_count == 1
    assert payload["items"][0]["style_concepts"] == ["vintage"]


def test_v3_cli_run_query_writes_outputs_and_candidate_dump(tmp_path: Path, monkeypatch) -> None:
    index_path = build_index_json(tmp_path)
    output_path = tmp_path / "results.csv"
    html_output_path = tmp_path / "results.html"
    candidate_output_path = tmp_path / "candidate.csv"
    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeQueryEncoder)
    monkeypatch.setattr(cli, "LuxiaV3QueryParser", StubParser)

    result = cli.run_query(
        index_path=str(index_path),
        query_text="black trousers",
        stage="mood_board",
        balance_score=0.0,
        user_uploaded_image=None,
        candidate_mode="union",
        symbolic_candidate_k=100,
        embedding_candidate_k=100,
        output_path=str(output_path),
        html_output_path=str(html_output_path),
        candidate_output_path=str(candidate_output_path),
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=20,
    )

    csv_lines = output_path.read_text(encoding="utf-8").splitlines()
    html = html_output_path.read_text(encoding="utf-8")
    candidate_lines = candidate_output_path.read_text(encoding="utf-8").splitlines()

    assert result.result_count == 2
    assert result.candidate_count == 2
    assert csv_lines[0] == "rank,image_id,score,brand,season_group,file_path,item_assignments,score_breakdown,match_reasons,explanation"
    assert "look-1" in csv_lines[1]
    assert "V3 Retrieval Preview" in html
    assert "candidate_mode: union" in html
    assert candidate_lines[0].startswith("query_text,stage,candidate_mode,rank,image_id")
    assert "in_symbolic_pool" in candidate_lines[0]
    assert "look-1" in candidate_lines[1]


def test_v3_cli_run_query_set_writes_manifest_and_candidate_judgments(tmp_path: Path, monkeypatch) -> None:
    index_path = build_index_json(tmp_path)
    queries_path = tmp_path / "queries.csv"
    queries_path.write_text(
        "\n".join(
            [
                "query_id,query_text,stage,query_type,expected_failure_type",
                "q001,black trousers,mood_board,color_sensitive,item_color_ambiguity",
            ]
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "eval"
    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeQueryEncoder)
    monkeypatch.setattr(cli, "LuxiaV3QueryParser", StubParser)

    result = cli.run_query_set(
        index_path=str(index_path),
        queries_path=str(queries_path),
        output_dir=str(output_dir),
        candidate_mode="union",
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=20,
        symbolic_candidate_k=100,
        embedding_candidate_k=100,
    )

    manifest_path = output_dir / "query_manifest.csv"
    candidate_judgments_path = output_dir / "candidate_judgments.csv"

    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    with open(candidate_judgments_path, newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.DictReader(handle))

    assert result.query_count == 1
    assert result.candidate_count == 2
    assert manifest_rows[0]["candidate_mode"] == "union"
    assert candidate_rows[0]["candidate_mode"] == "union"
    assert candidate_rows[0]["in_symbolic_pool"] in {"true", "false"}
    assert candidate_rows[0]["in_embedding_pool"] in {"true", "false"}


def test_v3_cli_compare_query_modes_writes_dashboard_and_mode_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    index_path = build_index_json(tmp_path)
    output_dir = tmp_path / "compare"
    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeQueryEncoder)
    monkeypatch.setattr(cli, "LuxiaV3QueryParser", StubParser)

    result = cli.compare_query_modes(
        index_path=str(index_path),
        query_text="black trousers",
        stage="mood_board",
        balance_score=0.0,
        user_uploaded_image=None,
        output_dir=str(output_dir),
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=20,
        symbolic_candidate_k=100,
        embedding_candidate_k=100,
    )

    dashboard_html = Path(result.dashboard_path).read_text(encoding="utf-8")

    assert result.mode_count == 3
    assert "V3 Mode Comparison" in dashboard_html
    assert "symbolic_only" in dashboard_html
    assert "embedding_only" in dashboard_html
    assert "union" in dashboard_html
    assert (output_dir / "symbolic_only.html").exists()
    assert (output_dir / "embedding_only.html").exists()
    assert (output_dir / "union.html").exists()
    assert (output_dir / "symbolic_only_candidates.csv").exists()
    assert (output_dir / "embedding_only_candidates.csv").exists()
    assert (output_dir / "union_candidates.csv").exists()
