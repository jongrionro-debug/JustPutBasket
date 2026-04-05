"""Orchestration helpers for V2 archive tagging and canonicalization."""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import sys
from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Protocol

from switch_query.tagging.preprocessing import (
    BlankTagger,
    CanonicalMappingRow,
    InventoryRow,
    NormalizedTagRow,
    RawTagRow,
    SampleRow,
    SubprocessJsonTagger,
    apply_canonical_mappings,
    build_frequency_rows,
    build_full_manifest,
    build_image_inventory,
    build_sample_manifest,
    _season_label_from_group,
    read_canonical_mappings,
    run_rough_tagging,
    seed_canonical_mappings,
    write_csv,
)
from switch_query.tagging.openai_batch_tagger import (
    OpenAIBatchTransport,
    build_batch_input_jsonl,
    collect_batch_results,
    ensure_uploaded_image_files,
    retrieve_batch_job,
    submit_batch,
)
from switch_query.tagging.openai_vlm_tagger import OpenAIJsonTagger, parse_openai_tag_response_text


DEFAULT_OUTPUT_ROOT = "tmp/v2_preprocessing"
DEFAULT_LOCAL_VLM_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
DEFAULT_OPENAI_VISION_MODEL = "gpt-4.1-mini"
DATASET_SLUG_COMPONENTS = 3
SLUG_SANITIZE_RE = re.compile(r"[^a-z0-9_-]+")
RAW_FEATURES = (
    "category",
    "silhouette",
    "color",
    "material",
    "pattern",
    "texture",
    "mood",
    "season",
    "era",
    "detail",
)


class ImageTagger(Protocol):
    def tag_image(self, sample: SampleRow):
        """Return a structured tag for one sample row."""


ProgressCallback = Callable[[int, int, int, int], None]


@dataclass(slots=True)
class V2PreprocessingPaths:
    dataset_root: str
    dataset_slug: str
    root_dir: str
    inventory_path: str
    sample_manifest_path: str
    raw_tags_sample_path: str
    frequency_sample_path: str
    canonical_mapping_draft_path: str
    raw_output_logs_sample_dir: str
    raw_tags_full_path: str
    frequency_full_path: str
    canonical_mapping_final_path: str
    raw_output_logs_full_dir: str
    normalized_tags_path: str
    image_file_ids_full_path: str
    batch_input_full_path: str
    batch_job_full_path: str
    batch_output_full_path: str
    batch_errors_full_path: str


@dataclass(slots=True)
class InventoryStageResult:
    paths: V2PreprocessingPaths
    inventory_count: int
    brand_count: int
    source_types: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TaggingStageSummary:
    row_count: int
    review_needed_count: int
    blank_caption_count: int
    blank_feature_counts: dict[str, int] = field(default_factory=dict)
    invalid_log_count: int = 0


@dataclass(slots=True)
class SampleStageResult:
    paths: V2PreprocessingPaths
    inventory_count: int
    sample_count: int
    raw_tag_count: int
    frequency_count: int
    canonical_draft_count: int
    tagging_summary: TaggingStageSummary


@dataclass(slots=True)
class FullTagStageResult:
    paths: V2PreprocessingPaths
    inventory_count: int
    raw_tag_count: int
    frequency_count: int
    tagging_summary: TaggingStageSummary


@dataclass(slots=True)
class NormalizeStageResult:
    paths: V2PreprocessingPaths
    raw_tag_count: int
    normalized_count: int
    rows_with_any_canonical_value: int


@dataclass(slots=True)
class SubmittedBatchResult:
    paths: V2PreprocessingPaths
    job_id: str
    input_file_id: str
    output_file_id: str | None
    error_file_id: str | None
    submitted_row_count: int


@dataclass(slots=True)
class CompletedBatchResult:
    paths: V2PreprocessingPaths
    job_id: str
    status: str
    raw_tag_count: int
    frequency_count: int
    invalid_row_count: int
    review_needed_count: int


@dataclass(slots=True)
class RecoveredFullTagResult:
    paths: V2PreprocessingPaths
    inventory_count: int
    recovered_count: int
    frequency_count: int
    available_parsed_log_count: int
    duplicate_stem_count: int
    tagging_summary: TaggingStageSummary


def build_preprocessing_paths(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
) -> V2PreprocessingPaths:
    resolved_root = Path(dataset_root).resolve()
    slug = dataset_slug or _derive_dataset_slug(resolved_root)
    root_dir = Path(output_root).resolve() / slug
    return V2PreprocessingPaths(
        dataset_root=str(resolved_root),
        dataset_slug=slug,
        root_dir=str(root_dir),
        inventory_path=str(root_dir / "inventory.csv"),
        sample_manifest_path=str(root_dir / "sample_manifest.csv"),
        raw_tags_sample_path=str(root_dir / "raw_tags_sample.csv"),
        frequency_sample_path=str(root_dir / "frequency_sample.csv"),
        canonical_mapping_draft_path=str(root_dir / "canonical_mapping_draft.csv"),
        raw_output_logs_sample_dir=str(root_dir / "raw_output_logs_sample"),
        raw_tags_full_path=str(root_dir / "raw_tags_full.csv"),
        frequency_full_path=str(root_dir / "frequency_full.csv"),
        canonical_mapping_final_path=str(root_dir / "canonical_mapping_final.csv"),
        raw_output_logs_full_dir=str(root_dir / "raw_output_logs_full"),
        normalized_tags_path=str(root_dir / "normalized_tags.csv"),
        image_file_ids_full_path=str(root_dir / "image_file_ids_full.csv"),
        batch_input_full_path=str(root_dir / "batch_input_full.jsonl"),
        batch_job_full_path=str(root_dir / "batch_job_full.json"),
        batch_output_full_path=str(root_dir / "batch_output_full.jsonl"),
        batch_errors_full_path=str(root_dir / "batch_errors_full.jsonl"),
    )


def run_inventory_stage(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
) -> InventoryStageResult:
    paths = build_preprocessing_paths(
        dataset_root,
        output_root=output_root,
        dataset_slug=dataset_slug,
    )
    inventory = build_image_inventory(dataset_root)
    if not inventory:
        raise ValueError(f"No collection images found under {dataset_root}")
    write_csv(paths.inventory_path, inventory)
    return InventoryStageResult(
        paths=paths,
        inventory_count=len(inventory),
        brand_count=len({row.brand for row in inventory}),
        source_types=sorted({row.source_type for row in inventory}),
    )


def run_sample_first_stage(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
    tagger_type: str = "openai-sync",
    model_name: str = DEFAULT_LOCAL_VLM_MODEL,
    raw_output_log_dir: str | None = None,
    tagger_command: list[str] | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    limit: int | None = None,
    tagger: ImageTagger | None = None,
) -> SampleStageResult:
    paths = build_preprocessing_paths(
        dataset_root,
        output_root=output_root,
        dataset_slug=dataset_slug,
    )
    inventory = build_image_inventory(dataset_root)
    if not inventory:
        raise ValueError(f"No collection images found under {dataset_root}")
    write_csv(paths.inventory_path, inventory)

    sample_manifest = build_sample_manifest(inventory)
    write_csv(paths.sample_manifest_path, sample_manifest)

    resolved_log_dir = raw_output_log_dir or paths.raw_output_logs_sample_dir
    resolved_tagger = tagger or build_tagger(
        tagger_type=tagger_type,
        model_name=model_name,
        raw_output_log_dir=resolved_log_dir,
        tagger_command=tagger_command,
        api_key_env=api_key_env,
    )
    raw_rows = run_rough_tagging(sample_manifest, resolved_tagger, limit=limit)
    write_csv(paths.raw_tags_sample_path, raw_rows)

    frequency_rows = build_frequency_rows(raw_rows)
    write_csv(paths.frequency_sample_path, frequency_rows)

    canonical_draft = seed_canonical_mappings(frequency_rows)
    write_csv(paths.canonical_mapping_draft_path, canonical_draft)

    return SampleStageResult(
        paths=paths,
        inventory_count=len(inventory),
        sample_count=len(sample_manifest[: limit or len(sample_manifest)]),
        raw_tag_count=len(raw_rows),
        frequency_count=len(frequency_rows),
        canonical_draft_count=len(canonical_draft),
        tagging_summary=summarize_raw_tags(raw_rows, raw_output_log_dir=resolved_log_dir),
    )


def run_full_tag_stage(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
    tagger_type: str = "openai-sync",
    model_name: str = DEFAULT_LOCAL_VLM_MODEL,
    raw_output_log_dir: str | None = None,
    tagger_command: list[str] | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    limit: int | None = None,
    max_workers: int = 8,
    fail_fast: bool = False,
    progress_callback: ProgressCallback | None = None,
    tagger: ImageTagger | None = None,
) -> FullTagStageResult:
    paths = build_preprocessing_paths(
        dataset_root,
        output_root=output_root,
        dataset_slug=dataset_slug,
    )
    inventory = build_image_inventory(dataset_root)
    if not inventory:
        raise ValueError(f"No collection images found under {dataset_root}")
    write_csv(paths.inventory_path, inventory)

    manifest = build_full_manifest(inventory)
    resolved_log_dir = raw_output_log_dir or paths.raw_output_logs_full_dir
    resolved_tagger = tagger or build_tagger(
        tagger_type=tagger_type,
        model_name=model_name,
        raw_output_log_dir=resolved_log_dir,
        tagger_command=tagger_command,
        api_key_env=api_key_env,
    )
    if tagger_type in {"openai", "openai-sync"}:
        raw_rows = run_parallel_openai_tagging(
            manifest,
            resolved_tagger,
            limit=limit,
            max_workers=max_workers,
            fail_fast=fail_fast,
            progress_callback=progress_callback,
        )
    else:
        raw_rows = run_rough_tagging(manifest, resolved_tagger, limit=limit)
    write_csv(paths.raw_tags_full_path, raw_rows)

    frequency_rows = build_frequency_rows(raw_rows)
    write_csv(paths.frequency_full_path, frequency_rows)

    return FullTagStageResult(
        paths=paths,
        inventory_count=len(inventory),
        raw_tag_count=len(raw_rows),
        frequency_count=len(frequency_rows),
        tagging_summary=summarize_raw_tags(raw_rows, raw_output_log_dir=resolved_log_dir),
    )


def run_normalize_stage(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
    raw_tags_path: str | None = None,
    canonical_mappings_path: str | None = None,
) -> NormalizeStageResult:
    paths = build_preprocessing_paths(
        dataset_root,
        output_root=output_root,
        dataset_slug=dataset_slug,
    )
    resolved_raw_tags_path = raw_tags_path or paths.raw_tags_full_path
    resolved_mappings_path = canonical_mappings_path or paths.canonical_mapping_final_path
    if not Path(resolved_raw_tags_path).exists():
        raise ValueError(f"Raw tags file not found: {resolved_raw_tags_path}")
    if not Path(resolved_mappings_path).exists():
        raise ValueError(f"Canonical mappings file not found: {resolved_mappings_path}")

    raw_rows = read_raw_tag_rows(resolved_raw_tags_path)
    mappings = read_canonical_mappings(resolved_mappings_path)
    normalized_rows = apply_canonical_mappings(raw_rows, mappings)
    write_csv(paths.normalized_tags_path, normalized_rows)

    return NormalizeStageResult(
        paths=paths,
        raw_tag_count=len(raw_rows),
        normalized_count=len(normalized_rows),
        rows_with_any_canonical_value=sum(
            1
            for row in normalized_rows
            if any(getattr(row, f"canonical_{feature}").strip() for feature in RAW_FEATURES)
        ),
    )


def build_tagger(
    *,
    tagger_type: str,
    model_name: str,
    raw_output_log_dir: str | None = None,
    tagger_command: list[str] | None = None,
    api_key_env: str = "OPENAI_API_KEY",
) -> ImageTagger:
    if tagger_type == "blank":
        return BlankTagger()
    if tagger_type in {"openai", "openai-sync"}:
        return OpenAIJsonTagger(
            model=(
                DEFAULT_OPENAI_VISION_MODEL
                if not model_name or model_name == DEFAULT_LOCAL_VLM_MODEL
                else model_name
            ),
            api_key_env=api_key_env,
            raw_output_log_dir=raw_output_log_dir,
        )
    if tagger_type == "subprocess":
        if not tagger_command:
            raise ValueError("tagger_command is required when tagger_type='subprocess'")
        return SubprocessJsonTagger(tagger_command)
    if tagger_type != "mlx_vlm":
        raise ValueError(f"Unsupported tagger_type: {tagger_type}")
    base_command = [
        sys.executable,
        "-m",
        "switch_query.tagging.local_vlm_tagger",
        "--model",
        model_name,
    ]
    if raw_output_log_dir:
        base_command.extend(["--raw-output-log-dir", raw_output_log_dir])
    return SubprocessJsonTagger(base_command)


def recover_full_tag_from_logs(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
    raw_output_log_dir: str | None = None,
    limit: int | None = None,
) -> RecoveredFullTagResult:
    paths = build_preprocessing_paths(
        dataset_root,
        output_root=output_root,
        dataset_slug=dataset_slug,
    )
    inventory = build_image_inventory(dataset_root)
    if not inventory:
        raise ValueError(f"No collection images found under {dataset_root}")
    write_csv(paths.inventory_path, inventory)

    log_dir = Path(raw_output_log_dir or paths.raw_output_logs_full_dir)
    if not log_dir.exists():
        raise ValueError(f"Raw output log directory not found: {log_dir}")

    latest_logs_by_stem, duplicate_stem_count = _collect_latest_parsed_logs(log_dir)
    manifest = build_full_manifest(inventory)
    if limit is not None:
        manifest = manifest[:limit]

    raw_rows: list[RawTagRow] = []
    for sample in manifest:
        stem = Path(sample.filename).stem
        log_path = latest_logs_by_stem.get(stem)
        if log_path is None:
            continue
        payload = parse_openai_tag_response_text(
            log_path.read_text(encoding="utf-8"),
            image_path=sample.file_path,
        )
        row = RawTagRow(
            **asdict(sample),
            caption=str(payload.get("caption", "")).strip(),
            raw_category=str(payload.get("category", "")).strip(),
            raw_silhouette=str(payload.get("silhouette", "")).strip(),
            raw_color=str(payload.get("color", "")).strip(),
            raw_material=str(payload.get("material", "")).strip(),
            raw_pattern=str(payload.get("pattern", "")).strip(),
            raw_texture=str(payload.get("texture", "")).strip(),
            raw_mood=str(payload.get("mood", "")).strip(),
            raw_season=_season_label_from_group(sample.season_group),
            raw_era=str(payload.get("era", "")).strip(),
            raw_detail=str(payload.get("detail", "")).strip(),
            review_needed="true" if bool(payload.get("review_needed", False)) else "false",
            confidence_note=str(payload.get("confidence_note", "")).strip(),
        )
        raw_rows.append(row)

    write_csv(paths.raw_tags_full_path, raw_rows)
    frequency_rows = build_frequency_rows(raw_rows)
    write_csv(paths.frequency_full_path, frequency_rows)

    return RecoveredFullTagResult(
        paths=paths,
        inventory_count=len(inventory),
        recovered_count=len(raw_rows),
        frequency_count=len(frequency_rows),
        available_parsed_log_count=len(latest_logs_by_stem),
        duplicate_stem_count=duplicate_stem_count,
        tagging_summary=summarize_raw_tags(raw_rows, raw_output_log_dir=str(log_dir)),
    )


def run_parallel_openai_tagging(
    manifest: list[SampleRow],
    tagger: ImageTagger,
    *,
    limit: int | None = None,
    max_workers: int = 8,
    fail_fast: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> list[RawTagRow]:
    if limit is None:
        limit = len(manifest)
    selected = manifest[:limit]
    total = len(selected)
    if total == 0:
        return []
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")

    rows_by_index: dict[int, RawTagRow] = {}
    success_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_tag_single_sample, index, sample, tagger): (index, sample)
            for index, sample in enumerate(selected)
        }
        completed = 0
        for future in as_completed(futures):
            index, sample = futures[future]
            try:
                row = future.result()
                if row.review_needed.lower() == "true":
                    error_count += 1
                else:
                    success_count += 1
            except Exception as exc:
                if fail_fast:
                    raise
                row = _build_error_raw_tag_row(sample, exc)
                error_count += 1
            rows_by_index[index] = row
            completed += 1
            if progress_callback:
                progress_callback(completed, total, success_count, error_count)

    return [rows_by_index[index] for index in range(total)]


def _tag_single_sample(index: int, sample: SampleRow, tagger: ImageTagger) -> RawTagRow:
    tagged = tagger.tag_image(sample)
    return RawTagRow(
        **asdict(sample),
        caption=tagged.caption,
        raw_category=tagged.category,
        raw_silhouette=tagged.silhouette,
        raw_color=tagged.color,
        raw_material=tagged.material,
        raw_pattern=tagged.pattern,
        raw_texture=tagged.texture,
        raw_mood=tagged.mood,
        raw_season=_season_label_from_group(sample.season_group),
        raw_era=tagged.era,
        raw_detail=tagged.detail,
        review_needed="true" if tagged.review_needed else "false",
        confidence_note=tagged.confidence_note,
    )


def _build_error_raw_tag_row(sample: SampleRow, exc: Exception) -> RawTagRow:
    error_text = str(exc).strip() or exc.__class__.__name__
    if len(error_text) > 240:
        error_text = error_text[:237] + "..."
    return RawTagRow(
        **asdict(sample),
        caption="",
        raw_category="",
        raw_silhouette="",
        raw_color="",
        raw_material="",
        raw_pattern="",
        raw_texture="",
        raw_mood="",
        raw_season=_season_label_from_group(sample.season_group),
        raw_era="",
        raw_detail="",
        review_needed="true",
        confidence_note=f"Tagging failed: {error_text}",
    )


def _collect_latest_parsed_logs(log_dir: Path) -> tuple[dict[str, Path], int]:
    latest_by_stem: dict[str, Path] = {}
    duplicate_stem_count = 0
    for path in sorted(log_dir.glob("*-parsed.txt")):
        stem = path.name.rsplit("-", 2)[0]
        if stem in latest_by_stem:
            duplicate_stem_count += 1
            previous = latest_by_stem[stem]
            if path.stat().st_mtime >= previous.stat().st_mtime:
                latest_by_stem[stem] = path
        else:
            latest_by_stem[stem] = path
    return latest_by_stem, duplicate_stem_count


def submit_full_tag_batch(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
    model_name: str = DEFAULT_OPENAI_VISION_MODEL,
    api_key_env: str = "OPENAI_API_KEY",
    limit: int | None = None,
    transport: OpenAIBatchTransport | None = None,
) -> SubmittedBatchResult:
    paths = build_preprocessing_paths(
        dataset_root,
        output_root=output_root,
        dataset_slug=dataset_slug,
    )
    inventory = build_image_inventory(dataset_root)
    if not inventory:
        raise ValueError(f"No collection images found under {dataset_root}")
    write_csv(paths.inventory_path, inventory)
    manifest = build_full_manifest(inventory)
    samples = manifest[: limit or len(manifest)]
    image_file_ids = ensure_uploaded_image_files(
        samples,
        cache_path=paths.image_file_ids_full_path,
        api_key_env=api_key_env,
        transport=transport,
    )
    submitted_row_count = build_batch_input_jsonl(
        samples,
        output_path=paths.batch_input_full_path,
        model=model_name,
        image_file_ids=image_file_ids,
    )
    job = submit_batch(
        batch_input_path=paths.batch_input_full_path,
        api_key_env=api_key_env,
        transport=transport,
    )
    Path(paths.batch_job_full_path).parent.mkdir(parents=True, exist_ok=True)
    Path(paths.batch_job_full_path).write_text(
        json.dumps(
            {
                "job_id": job.job_id,
                "input_file_id": job.input_file_id,
                "output_file_id": job.output_file_id,
                "error_file_id": job.error_file_id,
                "status": job.status,
                "submitted_row_count": submitted_row_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return SubmittedBatchResult(
        paths=paths,
        job_id=job.job_id,
        input_file_id=job.input_file_id,
        output_file_id=job.output_file_id,
        error_file_id=job.error_file_id,
        submitted_row_count=submitted_row_count,
    )


def collect_full_tag_batch(
    dataset_root: str,
    *,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    dataset_slug: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    transport: OpenAIBatchTransport | None = None,
    job_id: str | None = None,
) -> CompletedBatchResult:
    paths = build_preprocessing_paths(
        dataset_root,
        output_root=output_root,
        dataset_slug=dataset_slug,
    )
    if job_id:
        current_job_id = job_id
    else:
        if not Path(paths.batch_job_full_path).exists():
            raise ValueError(f"Batch job metadata not found: {paths.batch_job_full_path}")
        job_payload = json.loads(Path(paths.batch_job_full_path).read_text(encoding="utf-8"))
        current_job_id = str(job_payload.get("job_id", "")).strip()
    if not current_job_id:
        raise ValueError("Batch job id is required to collect full tag batch results")

    inventory = build_image_inventory(dataset_root)
    manifest = build_full_manifest(inventory)
    samples_by_id = {sample.image_id: sample for sample in manifest}
    job = retrieve_batch_job(current_job_id, api_key_env=api_key_env, transport=transport)
    if job.status != "completed":
        Path(paths.batch_job_full_path).write_text(
            json.dumps(
                {
                    "job_id": job.job_id,
                    "input_file_id": job.input_file_id,
                    "output_file_id": job.output_file_id,
                    "error_file_id": job.error_file_id,
                    "status": job.status,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise ValueError(f"Batch job {job.job_id} is not completed yet (status={job.status})")

    raw_rows, invalid_count = collect_batch_results(
        job=job,
        samples_by_id=samples_by_id,
        output_path=paths.batch_output_full_path,
        errors_path=paths.batch_errors_full_path,
        raw_output_log_dir=paths.raw_output_logs_full_dir,
        api_key_env=api_key_env,
        transport=transport,
    )
    write_csv(paths.raw_tags_full_path, raw_rows)
    frequency_rows = build_frequency_rows(raw_rows)
    write_csv(paths.frequency_full_path, frequency_rows)
    Path(paths.batch_job_full_path).write_text(
        json.dumps(
            {
                "job_id": job.job_id,
                "input_file_id": job.input_file_id,
                "output_file_id": job.output_file_id,
                "error_file_id": job.error_file_id,
                "status": job.status,
                "raw_tag_count": len(raw_rows),
                "invalid_row_count": invalid_count,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return CompletedBatchResult(
        paths=paths,
        job_id=job.job_id,
        status=job.status,
        raw_tag_count=len(raw_rows),
        frequency_count=len(frequency_rows),
        invalid_row_count=invalid_count,
        review_needed_count=sum(1 for row in raw_rows if row.review_needed.lower() == "true"),
    )


def summarize_raw_tags(
    rows: list[RawTagRow],
    *,
    raw_output_log_dir: str | None = None,
) -> TaggingStageSummary:
    blank_feature_counts = {
        feature: sum(1 for row in rows if not getattr(row, f"raw_{feature}", "").strip())
        for feature in RAW_FEATURES
    }
    invalid_log_count = 0
    if raw_output_log_dir and Path(raw_output_log_dir).exists():
        invalid_log_count = len(list(Path(raw_output_log_dir).glob("*-invalid.txt")))
    return TaggingStageSummary(
        row_count=len(rows),
        review_needed_count=sum(1 for row in rows if row.review_needed.lower() == "true"),
        blank_caption_count=sum(1 for row in rows if not row.caption.strip()),
        blank_feature_counts=blank_feature_counts,
        invalid_log_count=invalid_log_count,
    )


def read_raw_tag_rows(path: str) -> list[RawTagRow]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [RawTagRow(**row) for row in reader]


def read_normalized_tag_rows(path: str) -> list[NormalizedTagRow]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [NormalizedTagRow(**row) for row in reader]


def _derive_dataset_slug(dataset_root: Path) -> str:
    parts = [part for part in dataset_root.parts if part not in {dataset_root.anchor, ""}]
    if not parts:
        return "dataset"
    selected = parts[-DATASET_SLUG_COMPONENTS:]
    slug_parts = [_sanitize_slug_component(part) for part in selected]
    return "__".join(part for part in slug_parts if part) or "dataset"


def _sanitize_slug_component(value: str) -> str:
    normalized = value.lower().strip().replace(" ", "-")
    normalized = SLUG_SANITIZE_RE.sub("-", normalized)
    normalized = normalized.strip("-")
    return normalized or "dataset"
