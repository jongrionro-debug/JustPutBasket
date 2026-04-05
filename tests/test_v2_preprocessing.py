from __future__ import annotations

import json
import time
from pathlib import Path

from switch_query.tagging import write_csv
from switch_query.tagging.preprocessing import CanonicalMappingRow, TaggingResult
from switch_query.v2.preprocessing import (
    build_preprocessing_paths,
    collect_full_tag_batch,
    recover_full_tag_from_logs,
    read_raw_tag_rows,
    run_full_tag_stage,
    run_inventory_stage,
    run_normalize_stage,
    run_parallel_openai_tagging,
    run_sample_first_stage,
    submit_full_tag_batch,
    summarize_raw_tags,
)


class FakeTagger:
    def tag_image(self, sample):
        if sample.brand == "alpha":
            return TaggingResult(
                caption="black wool coat",
                category="coat|trousers",
                silhouette="sharp tailoring",
                color="jet black",
                material="wool",
                pattern="solid",
                texture="smooth",
                mood="minimal but sharp",
                season="fall",
                era="modern",
                detail="long coat|wide leg trousers",
                review_needed=False,
                confidence_note="high",
            )
        return TaggingResult(
            caption="red silk dress",
            category="gown",
            silhouette="fluid",
            color="scarlet",
            material="silk",
            pattern="solid",
            texture="satin",
            mood="romantic",
            season="spring",
            era="vintage",
            detail="evening gown|heels",
            review_needed=True,
            confidence_note="check subtype",
        )


class FakeBatchTransport:
    def __init__(self) -> None:
        self.uploaded_paths: list[str] = []
        self.uploaded_purposes: list[str] = []
        self.created_batches: list[str] = []
        self.batch_jobs: dict[str, dict[str, object]] = {
            "batch-123": {
                "id": "batch-123",
                "input_file_id": "file-input-123",
                "output_file_id": "file-output-123",
                "error_file_id": "file-error-123",
                "status": "completed",
            }
        }

    def upload_file(self, path: str, *, purpose: str = "batch") -> dict[str, object]:
        self.uploaded_paths.append(path)
        self.uploaded_purposes.append(purpose)
        if purpose == "vision":
            stem = Path(path).stem
            return {"id": f"file-vision-{stem}", "purpose": purpose}
        return {"id": "file-input-123", "purpose": purpose}

    def create_batch(self, input_file_id: str) -> dict[str, object]:
        self.created_batches.append(input_file_id)
        return {
            "id": "batch-123",
            "input_file_id": input_file_id,
            "output_file_id": "file-output-123",
            "error_file_id": "file-error-123",
            "status": "validating",
        }

    def retrieve_batch(self, batch_id: str) -> dict[str, object]:
        return self.batch_jobs[batch_id]

    def download_file_text(self, file_id: str) -> str:
        if file_id == "file-output-123":
            return "\n".join(
                [
                    json.dumps(
                        {
                            "custom_id": "2026:spring-ready-to-wear:alpha:0001_a",
                            "response": {
                                "body": {
                                    "output_text": (
                                        '{"caption":"black wool coat","category":"coat|trousers",'
                                        '"silhouette":"sharp tailoring","color":"jet black","material":"wool",'
                                        '"pattern":"solid","texture":"smooth","mood":"minimal but sharp",'
                                        '"season":"fall","era":"modern","detail":"long coat|wide leg trousers",'
                                        '"review_needed":false,"confidence_note":"high"}'
                                    )
                                }
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "custom_id": "2026:spring-ready-to-wear:alpha:0002_b",
                            "response": {
                                "body": {
                                    "output_text": (
                                        '{"caption":"black wool coat","category":"coat|trousers",'
                                        '"silhouette":"sharp tailoring","color":"jet black","material":"wool",'
                                        '"pattern":"solid","texture":"smooth","mood":"minimal but sharp",'
                                        '"season":"fall","era":"modern","detail":"long coat|wide leg trousers",'
                                        '"review_needed":false,"confidence_note":"high"}'
                                    )
                                }
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "custom_id": "2026:spring-ready-to-wear:beta:0001_c",
                            "response": {
                                "body": {
                                    "output_text": (
                                        '{"caption":"red silk dress","category":"gown","silhouette":"fluid",'
                                        '"color":"scarlet","material":"silk","pattern":"solid","texture":"satin",'
                                        '"mood":"romantic","season":"spring","era":"vintage",'
                                        '"detail":"evening gown|heels","review_needed":true,'
                                        '"confidence_note":"check subtype"}'
                                    )
                                }
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "custom_id": "2026:spring-ready-to-wear:beta:0002_d",
                            "response": {"body": {"output_text": "not json"}},
                        }
                    ),
                ]
            )
        if file_id == "file-error-123":
            return ""
        raise AssertionError(f"Unexpected file id: {file_id}")


class SlowTagger:
    def tag_image(self, sample):
        if sample.filename.endswith("0001_a.jpg"):
            time.sleep(0.03)
        if sample.filename.endswith("0002_b.jpg"):
            time.sleep(0.01)
        return TaggingResult(
            caption=f"caption-{sample.filename}",
            category="coat",
            silhouette="tailored",
            color="black",
            review_needed=False,
            confidence_note="ok",
        )


class PartiallyFailingTagger:
    def tag_image(self, sample):
        if sample.filename.endswith("0002_b.jpg"):
            raise RuntimeError("temporary upstream error")
        return TaggingResult(
            caption=f"caption-{sample.filename}",
            category="coat",
            silhouette="tailored",
            color="black",
            review_needed=False,
            confidence_note="ok",
        )


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def write_parsed_log(log_dir: Path, stem: str, digest: str, payload: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"{stem}-{digest}-parsed.txt").write_text(payload, encoding="utf-8")


def build_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "data" / "2026" / "spring-ready-to-wear"
    create_image(dataset_root / "alpha" / "collection" / "0001_a.jpg")
    create_image(dataset_root / "alpha" / "collection" / "0002_b.jpg")
    create_image(dataset_root / "beta" / "collection" / "0001_c.jpg")
    create_image(dataset_root / "beta" / "collection" / "0002_d.jpg")
    create_image(dataset_root / "beta" / "lookbook" / "ignore.jpg")
    return dataset_root


def test_build_preprocessing_paths_uses_fixed_artifact_names(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)

    paths = build_preprocessing_paths(str(dataset_root), output_root=str(tmp_path / "tmp"))

    assert paths.dataset_slug == "data__2026__spring-ready-to-wear"
    assert paths.inventory_path.endswith("inventory.csv")
    assert paths.sample_manifest_path.endswith("sample_manifest.csv")
    assert paths.raw_tags_sample_path.endswith("raw_tags_sample.csv")
    assert paths.canonical_mapping_final_path.endswith("canonical_mapping_final.csv")
    assert paths.normalized_tags_path.endswith("normalized_tags.csv")
    assert paths.image_file_ids_full_path.endswith("image_file_ids_full.csv")


def test_run_inventory_stage_writes_inventory_csv(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)

    result = run_inventory_stage(str(dataset_root), output_root=str(tmp_path / "tmp"))

    assert result.inventory_count == 4
    assert result.brand_count == 2
    assert result.source_types == ["collection"]
    assert Path(result.paths.inventory_path).exists()


def test_run_sample_first_stage_writes_sample_outputs(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)

    result = run_sample_first_stage(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        tagger=FakeTagger(),
    )

    assert result.sample_count == 4
    assert result.raw_tag_count == 4
    assert result.frequency_count > 0
    assert result.canonical_draft_count == result.frequency_count
    assert result.tagging_summary.review_needed_count == 2
    assert Path(result.paths.sample_manifest_path).exists()
    assert Path(result.paths.raw_tags_sample_path).exists()
    assert Path(result.paths.frequency_sample_path).exists()
    assert Path(result.paths.canonical_mapping_draft_path).exists()


def test_run_full_tag_stage_writes_full_outputs(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)

    result = run_full_tag_stage(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        tagger=FakeTagger(),
    )

    assert result.inventory_count == 4
    assert result.raw_tag_count == 4
    assert result.frequency_count > 0
    assert result.tagging_summary.blank_caption_count == 0
    assert Path(result.paths.raw_tags_full_path).exists()
    assert Path(result.paths.frequency_full_path).exists()


def test_run_parallel_openai_tagging_preserves_manifest_order(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)
    result = run_inventory_stage(str(dataset_root), output_root=str(tmp_path / "tmp"))
    from switch_query.tagging.preprocessing import build_full_manifest, build_image_inventory

    manifest = build_full_manifest(build_image_inventory(str(dataset_root)))

    rows = run_parallel_openai_tagging(manifest, SlowTagger(), max_workers=4)

    assert [row.filename for row in rows] == [sample.filename for sample in manifest]
    assert rows[0].caption == "caption-0001_a.jpg"
    assert rows[1].caption == "caption-0002_b.jpg"


def test_run_full_tag_stage_writes_fallback_row_for_failed_samples(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)

    result = run_full_tag_stage(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        tagger=PartiallyFailingTagger(),
        max_workers=4,
    )

    rows = read_raw_tag_rows(result.paths.raw_tags_full_path)
    failed = next(row for row in rows if row.filename == "0002_b.jpg")
    assert failed.review_needed == "true"
    assert "Tagging failed:" in failed.confidence_note
    assert result.tagging_summary.review_needed_count == 1


def test_submit_full_tag_batch_writes_input_and_job_metadata(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)
    transport = FakeBatchTransport()

    result = submit_full_tag_batch(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        transport=transport,
    )

    assert result.job_id == "batch-123"
    assert result.input_file_id == "file-input-123"
    assert result.submitted_row_count == 4
    assert Path(result.paths.batch_input_full_path).exists()
    assert Path(result.paths.image_file_ids_full_path).exists()
    job_payload = json.loads(Path(result.paths.batch_job_full_path).read_text(encoding="utf-8"))
    assert job_payload["job_id"] == "batch-123"
    assert transport.uploaded_purposes.count("vision") == 4
    assert transport.uploaded_purposes[-1] == "batch"
    assert transport.uploaded_paths[-1] == result.paths.batch_input_full_path
    cache_rows = Path(result.paths.image_file_ids_full_path).read_text(encoding="utf-8").splitlines()
    assert len(cache_rows) == 5


def test_collect_full_tag_batch_writes_raw_tags_and_frequency(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)
    transport = FakeBatchTransport()
    submit_result = submit_full_tag_batch(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        transport=transport,
    )

    result = collect_full_tag_batch(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        transport=transport,
        job_id=submit_result.job_id,
    )

    assert result.status == "completed"
    assert result.raw_tag_count == 4
    assert result.frequency_count > 0
    assert result.invalid_row_count == 1
    assert result.review_needed_count == 2
    assert Path(result.paths.raw_tags_full_path).exists()
    assert Path(result.paths.frequency_full_path).exists()
    assert Path(result.paths.batch_output_full_path).exists()
    assert Path(result.paths.batch_errors_full_path).exists()


def test_submit_full_tag_batch_reuses_cached_image_file_ids(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)
    first_transport = FakeBatchTransport()
    result = submit_full_tag_batch(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        transport=first_transport,
    )
    second_transport = FakeBatchTransport()

    second_result = submit_full_tag_batch(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        transport=second_transport,
    )

    assert result.paths.image_file_ids_full_path == second_result.paths.image_file_ids_full_path
    assert second_transport.uploaded_purposes == ["batch"]


def test_run_normalize_stage_writes_normalized_tags_csv(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)
    sample_result = run_sample_first_stage(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        tagger=FakeTagger(),
    )
    run_full_tag_stage(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
        tagger=FakeTagger(),
    )
    mappings = [
        CanonicalMappingRow("category", "coat", "coat", "review_needed", "", "draft"),
        CanonicalMappingRow("category", "trousers", "trousers", "review_needed", "", "draft"),
        CanonicalMappingRow("category", "dress", "gown", "parent_map", "", "approved"),
        CanonicalMappingRow("silhouette", "tailored", "sharp tailoring", "synonym", "", "approved"),
        CanonicalMappingRow("silhouette", "fluid", "fluid", "review_needed", "", "draft"),
        CanonicalMappingRow("color", "black", "jet black", "synonym", "", "approved"),
        CanonicalMappingRow("color", "red", "scarlet", "synonym", "", "approved"),
        CanonicalMappingRow("material", "wool", "wool", "review_needed", "", "draft"),
        CanonicalMappingRow("material", "silk", "silk", "review_needed", "", "draft"),
        CanonicalMappingRow("pattern", "solid", "solid", "review_needed", "", "draft"),
        CanonicalMappingRow("texture", "smooth", "smooth", "review_needed", "", "draft"),
        CanonicalMappingRow("texture", "satin", "satin", "review_needed", "", "draft"),
        CanonicalMappingRow("mood", "minimal|sharp", "minimal but sharp", "synonym", "", "approved"),
        CanonicalMappingRow("mood", "romantic", "romantic", "review_needed", "", "draft"),
        CanonicalMappingRow("season", "fall", "fall", "review_needed", "", "draft"),
        CanonicalMappingRow("season", "spring", "spring", "review_needed", "", "draft"),
        CanonicalMappingRow("era", "modern", "modern", "review_needed", "", "draft"),
        CanonicalMappingRow("era", "vintage", "vintage", "review_needed", "", "draft"),
        CanonicalMappingRow("detail", "long coat", "long coat", "review_needed", "", "draft"),
        CanonicalMappingRow("detail", "wide leg trousers", "wide leg trousers", "review_needed", "", "draft"),
        CanonicalMappingRow("detail", "gown", "evening gown", "parent_map", "", "approved"),
        CanonicalMappingRow("detail", "heels", "heels", "review_needed", "", "draft"),
    ]
    write_csv(sample_result.paths.canonical_mapping_final_path, mappings)

    result = run_normalize_stage(
        str(dataset_root),
        output_root=str(tmp_path / "tmp"),
    )

    normalized_csv = Path(result.paths.normalized_tags_path).read_text(encoding="utf-8")
    assert result.raw_tag_count == 4
    assert result.normalized_count == 4
    assert result.rows_with_any_canonical_value == 4
    assert "canonical_color" in normalized_csv
    assert "canonical_mood" in normalized_csv
    assert "black" in normalized_csv
    assert "minimal|sharp" in normalized_csv


def test_recover_full_tag_from_logs_rebuilds_subset_and_uses_latest_duplicate(tmp_path: Path) -> None:
    dataset_root = build_dataset(tmp_path)
    output_root = tmp_path / "tmp"
    paths = build_preprocessing_paths(str(dataset_root), output_root=str(output_root))
    log_dir = Path(paths.raw_output_logs_full_dir)
    write_parsed_log(
        log_dir,
        "0001_a",
        "old",
        '{"caption":"old caption","category":["coat"],"silhouette":"old","color":"black","review_needed":false,"confidence_note":"old"}',
    )
    write_parsed_log(
        log_dir,
        "0001_a",
        "new",
        '{"caption":"new caption","category":["coat"],"silhouette":"tailored","color":"black","review_needed":false,"confidence_note":"new"}',
    )
    write_parsed_log(
        log_dir,
        "0002_b",
        "one",
        '{"caption":"second caption","category":["dress"],"silhouette":"fluid","color":"red","review_needed":true,"confidence_note":"check"}',
    )

    result = recover_full_tag_from_logs(
        str(dataset_root),
        output_root=str(output_root),
        limit=2,
    )

    rows = read_raw_tag_rows(result.paths.raw_tags_full_path)
    assert result.recovered_count == 2
    assert result.available_parsed_log_count == 2
    assert result.duplicate_stem_count == 1
    assert rows[0].caption == "new caption"
    assert rows[0].raw_silhouette == "tailored"
    assert rows[1].review_needed == "true"


def test_summarize_raw_tags_counts_invalid_logs(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "look-1-invalid.txt").write_text("bad output", encoding="utf-8")
    rows = [
        type(
            "Row",
            (),
            {
                "review_needed": "true",
                "caption": "",
                "raw_category": "",
                "raw_silhouette": "tailored",
                "raw_color": "black",
                "raw_material": "",
                "raw_pattern": "",
                "raw_texture": "",
                "raw_mood": "",
                "raw_season": "",
                "raw_era": "",
                "raw_detail": "",
            },
        )()
    ]

    summary = summarize_raw_tags(rows, raw_output_log_dir=str(log_dir))

    assert summary.row_count == 1
    assert summary.review_needed_count == 1
    assert summary.blank_caption_count == 1
    assert summary.invalid_log_count == 1
    assert summary.blank_feature_counts["category"] == 1
