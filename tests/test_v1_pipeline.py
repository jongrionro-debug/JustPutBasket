from __future__ import annotations

from pathlib import Path

import pytest

from switch_query.tagging import InventoryRow
from switch_query.v1 import (
    InMemoryArchiveIndex,
    V1Pipeline,
    V1PipelineConfig,
    V1PipelineInput,
    average_dense_vectors,
    synthetic_reference_count,
)


class FakeEncoder:
    def __init__(self, text_vectors: dict[str, list[float]], image_vectors: dict[str, list[float]]) -> None:
        self.text_vectors = text_vectors
        self.image_vectors = image_vectors

    def encode_text(self, texts: list[str]) -> list[list[float]]:
        return [self.text_vectors[text] for text in texts]

    def encode_image(self, image_paths: list[str]) -> list[list[float]]:
        return [self.image_vectors[path] for path in image_paths]


class FakeImageGenerator:
    def __init__(self, generated_paths: list[str]) -> None:
        self.generated_paths = generated_paths

    def generate(
        self,
        query_text: str,
        count: int,
        *,
        query_id: str,
        balance_score: float,
    ) -> list[str]:
        return self.generated_paths[:count]


def build_inventory(tmp_path: Path) -> list[InventoryRow]:
    return [
        InventoryRow(
            image_id="look-1",
            file_path=str(tmp_path / "look-1.jpg"),
            season_group="spring-ready-to-wear",
            year="2026",
            brand="alpha",
            source_type="collection",
            filename="look-1.jpg",
        ),
        InventoryRow(
            image_id="look-2",
            file_path=str(tmp_path / "look-2.jpg"),
            season_group="spring-ready-to-wear",
            year="2026",
            brand="beta",
            source_type="collection",
            filename="look-2.jpg",
        ),
    ]


def test_synthetic_reference_count_matches_v1_policy() -> None:
    assert synthetic_reference_count(-0.8) == 4
    assert synthetic_reference_count(-0.2) == 3
    assert synthetic_reference_count(0.4) == 1


def test_average_dense_vectors_merges_generated_images() -> None:
    merged = average_dense_vectors([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    assert merged == pytest.approx([2 / 3, 2 / 3])


def test_build_archive_index_persists_inventory_metadata(tmp_path: Path) -> None:
    inventory = build_inventory(tmp_path)
    encoder = FakeEncoder(
        text_vectors={},
        image_vectors={
            inventory[0].file_path: [1.0, 0.0],
            inventory[1].file_path: [0.0, 1.0],
        },
    )
    pipeline = V1Pipeline(
        encoder=encoder,
        image_generator=FakeImageGenerator([]),
        index_store=InMemoryArchiveIndex(),
    )

    records = pipeline.build_archive_index(inventory)

    assert [record.image_id for record in records] == ["look-1", "look-2"]
    assert records[0].metadata["season_group"] == "spring-ready-to-wear"
    assert records[1].brand == "beta"


@pytest.mark.parametrize(
    ("retrieval_mode", "expected_score", "expected_image_score"),
    [
        ("text_only", 1.0, 0.0),
        ("image_only", 1.0, 1.0),
        ("fusion", 1.0, 1.0),
    ],
)
def test_pipeline_scores_text_image_and_fusion_modes(
    tmp_path: Path,
    retrieval_mode: str,
    expected_score: float,
    expected_image_score: float,
) -> None:
    inventory = build_inventory(tmp_path)
    generated_paths = ["gen-1.png"]
    encoder = FakeEncoder(
        text_vectors={"black tailored coat": [1.0, 0.0]},
        image_vectors={
            inventory[0].file_path: [1.0, 0.0],
            inventory[1].file_path: [0.0, 1.0],
            "gen-1.png": [1.0, 0.0],
        },
    )
    pipeline = V1Pipeline(
        encoder=encoder,
        image_generator=FakeImageGenerator(generated_paths),
        index_store=InMemoryArchiveIndex(),
        config=V1PipelineConfig(retrieval_mode=retrieval_mode, top_k=2),
    )
    pipeline.build_archive_index(inventory)

    output = pipeline.run(
        V1PipelineInput(
            query_id="q1",
            query_text="black tailored coat",
            balance_score=0.7,
        )
    )

    top_result = output.archive_results[0]
    assert top_result.image_id == "look-1"
    assert top_result.final_score == pytest.approx(expected_score)
    assert top_result.text_score == pytest.approx(1.0)
    assert top_result.image_score == pytest.approx(expected_image_score)


def test_pipeline_averages_multiple_generated_vectors_and_returns_score_logs(
    tmp_path: Path,
) -> None:
    inventory = build_inventory(tmp_path)
    generated_paths = ["gen-1.png", "gen-2.png", "gen-3.png"]
    encoder = FakeEncoder(
        text_vectors={"structured black coat": [1.0, 0.0]},
        image_vectors={
            inventory[0].file_path: [1.0, 0.0],
            inventory[1].file_path: [0.0, 1.0],
            "gen-1.png": [1.0, 0.0],
            "gen-2.png": [1.0, 0.0],
            "gen-3.png": [0.0, 1.0],
        },
    )
    pipeline = V1Pipeline(
        encoder=encoder,
        image_generator=FakeImageGenerator(generated_paths),
        index_store=InMemoryArchiveIndex(),
        config=V1PipelineConfig(top_k=2),
    )
    pipeline.build_archive_index(inventory)

    output = pipeline.run(
        V1PipelineInput(
            query_id="q-divergent",
            query_text="structured black coat",
            balance_score=-0.2,
        )
    )

    assert len(output.generated_references) == 3
    assert output.query_artifacts.merged_generated_image_vector == pytest.approx([2 / 3, 1 / 3])
    assert [result.rank for result in output.archive_results] == [1, 2]
    assert output.archive_results[0].final_score > output.archive_results[1].final_score
    assert output.archive_results[0].text_score == pytest.approx(1.0)
    assert output.archive_results[0].image_score == pytest.approx(0.894427, rel=1e-5)
