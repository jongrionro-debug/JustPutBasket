"""Synonym and preprocessing utilities for spring ready-to-wear v1."""

from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

RAW_FEATURES = ("category", "color", "material", "mood")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(slots=True)
class InventoryRow:
    image_id: str
    file_path: str
    season_group: str
    brand: str
    source_type: str
    filename: str


@dataclass(slots=True)
class SampleRow(InventoryRow):
    sample_reason: str


@dataclass(slots=True)
class RawTagRow(SampleRow):
    caption: str
    raw_category: str
    raw_color: str
    raw_material: str
    raw_mood: str
    review_needed: str
    confidence_note: str


@dataclass(slots=True)
class FrequencyRow:
    feature: str
    raw_value: str
    count: int
    example_image_ids: str


@dataclass(slots=True)
class CanonicalMappingRow:
    feature: str
    canonical: str
    variant: str
    mapping_type: str
    notes: str
    status: str


@dataclass(slots=True)
class NormalizedTagRow(RawTagRow):
    canonical_category: str
    canonical_color: str
    canonical_material: str
    canonical_mood: str


@dataclass(slots=True)
class RetrievalQuery:
    query_id: str
    query_text: str
    category: str = ""
    color: str = ""
    material: str = ""
    mood: str = ""


@dataclass(slots=True)
class RetrievalEvalRow:
    query_id: str
    query_text: str
    mode: str
    rank: int
    image_id: str
    brand: str
    score: float
    matched_fields: str
    comparison_notes: str


@dataclass(slots=True)
class TaggingResult:
    caption: str
    category: str = ""
    color: str = ""
    material: str = ""
    mood: str = ""
    review_needed: bool = False
    confidence_note: str = ""


class ImageTagger(Protocol):
    def tag_image(self, sample: SampleRow) -> TaggingResult:
        """Return a structured rough-tagging result for one image."""


class BlankTagger:
    """Safe fallback tagger that marks every sample for review."""

    def tag_image(self, sample: SampleRow) -> TaggingResult:
        return TaggingResult(
            caption="",
            review_needed=True,
            confidence_note="No local model runner configured",
        )


class SubprocessJsonTagger:
    """
    Adapter for any local model runner that prints a JSON object to stdout.

    The command receives one extra argument: the image path.
    Expected JSON keys: caption, category, color, material, mood, review_needed, confidence_note
    """

    def __init__(self, base_command: list[str]) -> None:
        if not base_command:
            raise ValueError("base_command must not be empty")
        self.base_command = list(base_command)

    def tag_image(self, sample: SampleRow) -> TaggingResult:
        try:
            completed = subprocess.run(
                [*self.base_command, sample.file_path],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise RuntimeError(
                f"Local tagger command failed for {sample.image_id}: {stderr or exc}"
            ) from exc
        payload = json.loads(completed.stdout)
        return TaggingResult(
            caption=str(payload.get("caption", "")).strip(),
            category=str(payload.get("category", "")).strip(),
            color=str(payload.get("color", "")).strip(),
            material=str(payload.get("material", "")).strip(),
            mood=str(payload.get("mood", "")).strip(),
            review_needed=bool(payload.get("review_needed", False)),
            confidence_note=str(payload.get("confidence_note", "")).strip(),
        )


def build_image_inventory(dataset_root: str | Path) -> list[InventoryRow]:
    root = Path(dataset_root)
    rows: list[InventoryRow] = []
    for brand_dir in sorted([path for path in root.iterdir() if path.is_dir()]):
        collection_dir = brand_dir / "collection"
        if not collection_dir.exists():
            continue
        files = sorted(
            [
                path
                for path in collection_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        for image_path in files:
            rows.append(
                InventoryRow(
                    image_id=f"{brand_dir.name}:{image_path.stem}",
                    file_path=str(image_path.resolve()),
                    season_group=root.name,
                    brand=brand_dir.name,
                    source_type="collection",
                    filename=image_path.name,
                )
            )
    return rows


def build_sample_manifest(inventory: list[InventoryRow]) -> list[SampleRow]:
    grouped: dict[str, list[InventoryRow]] = defaultdict(list)
    for row in inventory:
        grouped[row.brand].append(row)

    manifest: list[SampleRow] = []
    for brand in sorted(grouped):
        files = sorted(grouped[brand], key=lambda row: row.filename)
        first = files[0]
        last = files[-1]
        manifest.append(
            SampleRow(**asdict(first), sample_reason="brand_first")
        )
        if last.image_id != first.image_id:
            manifest.append(
                SampleRow(**asdict(last), sample_reason="brand_last")
            )
        else:
            manifest.append(
                SampleRow(**asdict(last), sample_reason="brand_last_duplicate")
            )
    return manifest


def run_rough_tagging(
    manifest: list[SampleRow],
    tagger: ImageTagger,
    limit: int | None = None,
) -> list[RawTagRow]:
    rows: list[RawTagRow] = []
    for sample in manifest[:limit]:
        tagged = tagger.tag_image(sample)
        rows.append(
            RawTagRow(
                **asdict(sample),
                caption=tagged.caption,
                raw_category=tagged.category,
                raw_color=tagged.color,
                raw_material=tagged.material,
                raw_mood=tagged.mood,
                review_needed="true" if tagged.review_needed else "false",
                confidence_note=tagged.confidence_note,
            )
        )
    return rows


def build_frequency_rows(raw_rows: list[RawTagRow]) -> list[FrequencyRow]:
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    counter: Counter[tuple[str, str]] = Counter()
    for row in raw_rows:
        for feature in RAW_FEATURES:
            value = getattr(row, f"raw_{feature}").strip()
            if not value:
                continue
            key = (feature, value)
            counter[key] += 1
            if len(examples[key]) < 5:
                examples[key].append(row.image_id)

    frequency_rows = [
        FrequencyRow(
            feature=feature,
            raw_value=value,
            count=count,
            example_image_ids="|".join(examples[(feature, value)]),
        )
        for (feature, value), count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    return frequency_rows


def seed_canonical_mappings(frequency_rows: list[FrequencyRow]) -> list[CanonicalMappingRow]:
    return [
        CanonicalMappingRow(
            feature=row.feature,
            canonical=row.raw_value,
            variant=row.raw_value,
            mapping_type="review_needed",
            notes="seeded from frequency table",
            status="draft",
        )
        for row in frequency_rows
    ]


def apply_canonical_mappings(
    raw_rows: list[RawTagRow],
    mappings: list[CanonicalMappingRow],
) -> list[NormalizedTagRow]:
    index = {
        (mapping.feature, mapping.variant): mapping.canonical
        for mapping in mappings
    }
    normalized_rows: list[NormalizedTagRow] = []
    for row in raw_rows:
        normalized_rows.append(
            NormalizedTagRow(
                **asdict(row),
                canonical_category=index.get(("category", row.raw_category), row.raw_category),
                canonical_color=index.get(("color", row.raw_color), row.raw_color),
                canonical_material=index.get(("material", row.raw_material), row.raw_material),
                canonical_mood=index.get(("mood", row.raw_mood), row.raw_mood),
            )
        )
    return normalized_rows


def build_default_queries(rows: list[NormalizedTagRow], max_queries: int = 12) -> list[RetrievalQuery]:
    category_counts = Counter(row.canonical_category for row in rows if row.canonical_category)
    mood_counts = Counter(row.canonical_mood for row in rows if row.canonical_mood)
    color_counts = Counter(row.canonical_color for row in rows if row.canonical_color)
    material_counts = Counter(row.canonical_material for row in rows if row.canonical_material)

    queries: list[RetrievalQuery] = []
    for idx, value in enumerate([item[0] for item in category_counts.most_common(3)], start=1):
        queries.append(
            RetrievalQuery(query_id=f"cat_{idx}", query_text=f"{value} look", category=value)
        )
    for idx, value in enumerate([item[0] for item in mood_counts.most_common(3)], start=1):
        queries.append(
            RetrievalQuery(query_id=f"mood_{idx}", query_text=f"{value} mood", mood=value)
        )
    for idx, value in enumerate([item[0] for item in color_counts.most_common(3)], start=1):
        queries.append(
            RetrievalQuery(query_id=f"color_{idx}", query_text=f"{value} look", color=value)
        )
    for idx, value in enumerate([item[0] for item in material_counts.most_common(3)], start=1):
        queries.append(
            RetrievalQuery(query_id=f"mat_{idx}", query_text=f"{value} look", material=value)
        )
    return queries[:max_queries]


def evaluate_retrieval(
    raw_rows: list[RawTagRow],
    normalized_rows: list[NormalizedTagRow],
    queries: list[RetrievalQuery] | None = None,
    top_k: int = 5,
) -> list[RetrievalEvalRow]:
    eval_queries = queries or build_default_queries(normalized_rows)
    raw_by_id = {row.image_id: row for row in raw_rows}
    normalized_by_id = {row.image_id: row for row in normalized_rows}

    logs: list[RetrievalEvalRow] = []
    for query in eval_queries:
        for mode, rows in (("raw", raw_by_id), ("canonical", normalized_by_id)):
            scored = []
            for row in rows.values():
                score, matched = _score_query_against_row(query, row, mode)
                if score > 0:
                    scored.append((score, matched, row))
            scored.sort(key=lambda item: (-item[0], item[2].image_id))
            for rank, (score, matched, row) in enumerate(scored[:top_k], start=1):
                logs.append(
                    RetrievalEvalRow(
                        query_id=query.query_id,
                        query_text=query.query_text,
                        mode=mode,
                        rank=rank,
                        image_id=row.image_id,
                        brand=row.brand,
                        score=score,
                        matched_fields="|".join(matched),
                        comparison_notes="",
                    )
                )
    return logs


def _score_query_against_row(
    query: RetrievalQuery,
    row: RawTagRow | NormalizedTagRow,
    mode: str,
) -> tuple[float, list[str]]:
    matched: list[str] = []
    score = 0.0
    prefix = "canonical_" if mode == "canonical" and isinstance(row, NormalizedTagRow) else "raw_"
    for feature in RAW_FEATURES:
        query_value = getattr(query, feature)
        row_value = getattr(row, f"{prefix}{feature}", "")
        if query_value and row_value and query_value == row_value:
            matched.append(feature)
            score += 1.0
    return score, matched


def write_csv(path: str | Path, rows: Iterable[object]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot write empty CSV without field names")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys())
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def read_canonical_mappings(path: str | Path) -> list[CanonicalMappingRow]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            CanonicalMappingRow(
                feature=row["feature"],
                canonical=row["canonical"],
                variant=row["variant"],
                mapping_type=row["mapping_type"],
                notes=row["notes"],
                status=row["status"],
            )
            for row in reader
        ]
