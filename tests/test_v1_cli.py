from __future__ import annotations

import json
from pathlib import Path

from switch_query.v1 import cli


class FakeEncoder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device or "cpu"

    def encode_text(self, texts):
        return [[0.0, 0.0] for _ in texts]

    def encode_image(self, image_paths):
        vectors = []
        for index, _path in enumerate(image_paths, start=1):
            vectors.append([float(index), 0.0])
        return vectors


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_build_archive_index_writes_json_index(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "data"
    create_image(
        dataset_root / "2026" / "spring-ready-to-wear" / "alpha" / "collection" / "0001.jpg"
    )
    create_image(
        dataset_root / "2026" / "spring-ready-to-wear" / "beta" / "collection" / "0002.jpg"
    )
    output_path = tmp_path / "archive_index.json"

    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeEncoder)

    result = cli.build_archive_index(
        dataset_root=str(dataset_root),
        output_path=str(output_path),
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.image_count == 2
    assert result.brand_count == 2
    assert result.model_name == "fake/siglip2"
    assert result.device == "cpu"
    assert payload[0]["metadata"]["season_group"] == "spring-ready-to-wear"
    assert payload[1]["brand"] == "beta"


def test_build_archive_index_rejects_empty_dataset(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeEncoder)

    try:
        cli.build_archive_index(
            dataset_root=str(tmp_path),
            output_path=str(tmp_path / "archive_index.json"),
            model_name="fake/siglip2",
            device="cpu",
            batch_size=4,
        )
    except ValueError as exc:
        assert "No collection images found" in str(exc)
    else:
        raise AssertionError("Expected build_archive_index to reject an empty dataset")


class FakeQueryEncoder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = config.device or "cpu"

    def encode_text(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def encode_image(self, image_paths):
        mapping = {
            "gen-1.png": [1.0, 0.0],
        }
        vectors = []
        for image_path in image_paths:
            vectors.append(mapping.get(image_path, [1.0, 0.0] if "look-1" in image_path else [0.0, 1.0]))
        return vectors


def test_run_query_writes_ranked_csv(tmp_path: Path, monkeypatch) -> None:
    index_path = tmp_path / "archive_index.json"
    index_path.write_text(
        json.dumps(
            [
                {
                    "image_id": "look-1",
                    "file_path": str(tmp_path / "look-1.jpg"),
                    "brand": "alpha",
                    "vector": [1.0, 0.0],
                    "metadata": {"season_group": "spring-ready-to-wear"},
                },
                {
                    "image_id": "look-2",
                    "file_path": str(tmp_path / "look-2.jpg"),
                    "brand": "beta",
                    "vector": [0.0, 1.0],
                    "metadata": {"season_group": "spring-ready-to-wear"},
                },
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "results.csv"

    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeQueryEncoder)

    result = cli.run_query(
        index_path=str(index_path),
        query_text="black tailored coat",
        query_id="query-1",
        balance_score=0.0,
        retrieval_mode="text_only",
        generated_image_paths=[],
        output_path=str(output_path),
        model_name="fake/siglip2",
        device="cpu",
        batch_size=4,
        top_k=2,
    )

    rows = output_path.read_text(encoding="utf-8").splitlines()

    assert result.result_count == 2
    assert result.generated_image_count == 0
    assert rows[0] == "query_id,query_text,rank,image_id,final_score,text_score,image_score,brand,file_path"
    assert "look-1" in rows[1]


def test_run_query_requires_generated_images_for_fusion_mode(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "SigLIP2Encoder", FakeQueryEncoder)

    try:
        cli.run_query(
            index_path=str(tmp_path / "archive_index.json"),
            query_text="black tailored coat",
            query_id="query-1",
            balance_score=0.0,
            retrieval_mode="fusion",
            generated_image_paths=[],
            output_path=None,
            model_name="fake/siglip2",
            device="cpu",
            batch_size=4,
            top_k=2,
        )
    except ValueError as exc:
        assert "generated_image_paths are required" in str(exc)
    else:
        raise AssertionError("Expected fusion mode to require generated images")
