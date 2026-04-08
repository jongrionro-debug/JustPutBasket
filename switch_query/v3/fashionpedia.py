"""Import helpers for converting Fashionpedia into V3 archive documents."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import tempfile
from urllib import request
import zipfile

from .concepts import normalize_token
from .models import V3ArchiveDocument, V3DocumentItem
from .preprocessing import write_archive_documents_jsonl

DEFAULT_FASHIONPEDIA_DATASET_ROOT = "data/external/fashionpedia"
DEFAULT_FASHIONPEDIA_OUTPUT_DIR = "tmp/v3_preprocessing/data__fashionpedia_full"
DEFAULT_FASHIONPEDIA_DOCUMENTS_PATH = (
    f"{DEFAULT_FASHIONPEDIA_OUTPUT_DIR}/item_enriched_documents_full.jsonl"
)
DEFAULT_FASHIONPEDIA_PROVENANCE_PATH = (
    f"{DEFAULT_FASHIONPEDIA_OUTPUT_DIR}/provenance_manifest.jsonl"
)

TRAIN_IMAGES_URL = "https://s3.amazonaws.com/ifashionist-dataset/images/train2020.zip"
VAL_TEST_IMAGES_URL = "https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip"
TRAIN_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_train2020.json"
)
VAL_ANNOTATIONS_URL = (
    "https://s3.amazonaws.com/ifashionist-dataset/annotations/instances_attributes_val2020.json"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RESOURCE_DIR = Path(__file__).resolve().parent / "data"
FASHIONPEDIA_IMPORT_NOTE = "fashionpedia_import"
FASHIONPEDIA_SPLITS = ("train", "val")
FEATURE_FIELDS = (
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


@dataclass(slots=True)
class FashionpediaFetchResult:
    dataset_root: str
    train_image_dir: str
    val_image_dir: str
    train_annotations_path: str
    val_annotations_path: str
    downloaded_file_count: int
    extracted_archive_count: int


@dataclass(slots=True)
class FashionpediaConvertResult:
    dataset_root: str
    documents_path: str
    provenance_path: str
    document_count: int
    manifest_record_count: int
    converted_image_count: int
    skipped_missing_image_count: int
    skipped_empty_item_count: int


def fetch_fashionpedia_dataset(
    *,
    dataset_root: str = DEFAULT_FASHIONPEDIA_DATASET_ROOT,
) -> FashionpediaFetchResult:
    root = Path(dataset_root).resolve()
    annotations_dir = root / "annotations"
    images_dir = root / "images"
    downloads_dir = root / "downloads"
    train_dir = images_dir / "train"
    val_dir = images_dir / "val"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    train_annotations_path = annotations_dir / "instances_attributes_train2020.json"
    val_annotations_path = annotations_dir / "instances_attributes_val2020.json"
    downloaded_count = 0
    downloaded_count += int(
        _download_if_missing(TRAIN_ANNOTATIONS_URL, train_annotations_path)
    )
    downloaded_count += int(_download_if_missing(VAL_ANNOTATIONS_URL, val_annotations_path))

    train_payload = _load_json(train_annotations_path)
    val_payload = _load_json(val_annotations_path)
    train_zip_path = downloads_dir / "train2020.zip"
    val_zip_path = downloads_dir / "val_test2020.zip"
    downloaded_count += int(_download_if_missing(TRAIN_IMAGES_URL, train_zip_path))
    downloaded_count += int(_download_if_missing(VAL_TEST_IMAGES_URL, val_zip_path))

    extracted_count = 0
    extracted_count += int(
        _extract_referenced_images(
            archive_path=train_zip_path,
            referenced_file_names={str(image["file_name"]) for image in train_payload.get("images", [])},
            target_dir=train_dir,
        )
    )
    extracted_count += int(
        _extract_referenced_images(
            archive_path=val_zip_path,
            referenced_file_names={str(image["file_name"]) for image in val_payload.get("images", [])},
            target_dir=val_dir,
        )
    )

    return FashionpediaFetchResult(
        dataset_root=str(root),
        train_image_dir=str(train_dir),
        val_image_dir=str(val_dir),
        train_annotations_path=str(train_annotations_path),
        val_annotations_path=str(val_annotations_path),
        downloaded_file_count=downloaded_count,
        extracted_archive_count=extracted_count,
    )


def convert_fashionpedia_to_v3_documents(
    *,
    dataset_root: str = DEFAULT_FASHIONPEDIA_DATASET_ROOT,
    output_path: str = DEFAULT_FASHIONPEDIA_DOCUMENTS_PATH,
    provenance_path: str = DEFAULT_FASHIONPEDIA_PROVENANCE_PATH,
) -> FashionpediaConvertResult:
    root = Path(dataset_root).resolve()
    category_map = _load_mapping_json("fashionpedia_category_map.json")
    attribute_map = _load_mapping_json("fashionpedia_attribute_map.json")

    documents: list[V3ArchiveDocument] = []
    manifest_records: list[dict[str, object]] = []
    converted_image_count = 0
    skipped_missing_image_count = 0
    skipped_empty_item_count = 0

    for split in FASHIONPEDIA_SPLITS:
        payload = _load_annotation_payload(root, split=split)
        licenses_by_id = {int(item["id"]): item for item in payload.get("licenses", [])}
        categories_by_id = {int(item["id"]): item for item in payload.get("categories", [])}
        attributes_by_id = {int(item["id"]): item for item in payload.get("attributes", [])}
        annotations_by_image_id: dict[int, list[dict[str, object]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image_id[int(annotation["image_id"])].append(annotation)

        for image in payload.get("images", []):
            image_id = int(image["id"])
            local_path = root / "images" / split / str(image["file_name"])
            license_info = licenses_by_id.get(int(image.get("license", 0)), {})
            manifest = {
                "image_id": f"fashionpedia:{split}:{image_id}",
                "fashionpedia_image_id": image_id,
                "split": split,
                "local_image_path": str(local_path),
                "original_url": str(image.get("original_url", "")),
                "license_id": int(license_info.get("id", 0)) if license_info else 0,
                "license_name": str(license_info.get("name", "")),
                "license_url": str(license_info.get("url", "")),
                "dropped_annotation_count": 0,
                "unmapped_category_count": 0,
                "unmapped_attribute_ids": [],
                "unmapped_attribute_names": [],
                "document_emitted": False,
                "status": "pending",
            }
            if not local_path.exists():
                manifest["status"] = "missing_image"
                skipped_missing_image_count += 1
                manifest_records.append(manifest)
                continue

            converted_items: list[V3DocumentItem] = []
            unmapped_attribute_ids: list[int] = []
            unmapped_attribute_names: list[str] = []
            dropped_annotation_count = 0
            unmapped_category_count = 0
            sorted_annotations = sorted(
                annotations_by_image_id.get(image_id, []),
                key=_annotation_sort_key,
            )

            for annotation in sorted_annotations:
                converted = _convert_annotation_to_item(
                    annotation=annotation,
                    split=split,
                    image_id=image_id,
                    item_index=len(converted_items) + 1,
                    categories_by_id=categories_by_id,
                    attributes_by_id=attributes_by_id,
                    category_map=category_map,
                    attribute_map=attribute_map,
                )
                if converted is None:
                    dropped_annotation_count += 1
                    category_id = int(annotation.get("category_id", 0))
                    if category_id in categories_by_id:
                        unmapped_category_count += 1
                    continue
                item, annotation_unmapped_ids, annotation_unmapped_names = converted
                converted_items.append(item)
                for attribute_id in annotation_unmapped_ids:
                    if attribute_id not in unmapped_attribute_ids:
                        unmapped_attribute_ids.append(attribute_id)
                for attribute_name in annotation_unmapped_names:
                    if attribute_name not in unmapped_attribute_names:
                        unmapped_attribute_names.append(attribute_name)

            manifest["dropped_annotation_count"] = dropped_annotation_count
            manifest["unmapped_category_count"] = unmapped_category_count
            manifest["unmapped_attribute_ids"] = unmapped_attribute_ids
            manifest["unmapped_attribute_names"] = unmapped_attribute_names

            if not converted_items:
                manifest["status"] = "no_mapped_items"
                skipped_empty_item_count += 1
                manifest_records.append(manifest)
                continue

            detail = _build_document_detail(converted_items)
            canonical_tags = _build_document_tags(converted_items, detail=detail)
            raw_tags = dict(canonical_tags)
            document = V3ArchiveDocument(
                image_id=f"fashionpedia:{split}:{image_id}",
                file_path=str(local_path),
                brand="fashionpedia",
                season_group="fashionpedia",
                canonical_tags=canonical_tags,
                raw_tags=raw_tags,
                detail=detail,
                items=converted_items,
                item_confidence=1.0,
                item_extraction_notes=[FASHIONPEDIA_IMPORT_NOTE, f"fashionpedia_split:{split}"],
            )
            documents.append(document)
            manifest["document_emitted"] = True
            manifest["status"] = "converted"
            converted_image_count += 1
            manifest_records.append(manifest)

    write_archive_documents_jsonl(output_path, documents)
    _write_jsonl(provenance_path, manifest_records)
    return FashionpediaConvertResult(
        dataset_root=str(root),
        documents_path=str(Path(output_path).resolve()),
        provenance_path=str(Path(provenance_path).resolve()),
        document_count=len(documents),
        manifest_record_count=len(manifest_records),
        converted_image_count=converted_image_count,
        skipped_missing_image_count=skipped_missing_image_count,
        skipped_empty_item_count=skipped_empty_item_count,
    )


def _load_annotation_payload(root: Path, *, split: str) -> dict[str, object]:
    if split not in FASHIONPEDIA_SPLITS:
        raise ValueError(f"Unsupported Fashionpedia split: {split}")
    file_name = f"instances_attributes_{split}2020.json"
    path = root / "annotations" / file_name
    if not path.exists():
        raise FileNotFoundError(f"Fashionpedia annotations not found: {path}")
    return _load_json(path)


def _load_mapping_json(file_name: str) -> dict[str, object]:
    path = RESOURCE_DIR / file_name
    return json.loads(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _download_if_missing(url: str, destination: Path) -> bool:
    if destination.exists():
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    with request.urlopen(url) as response, open(destination, "wb") as handle:
        shutil.copyfileobj(response, handle)
    return True


def _extract_referenced_images(
    *,
    archive_path: Path,
    referenced_file_names: set[str],
    target_dir: Path,
) -> bool:
    if target_dir.exists() and all((target_dir / file_name).exists() for file_name in referenced_file_names):
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(temp_root)
        available_files = {}
        for candidate in temp_root.rglob("*"):
            if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_path = candidate.relative_to(temp_root).as_posix()
            available_files[relative_path] = candidate
            available_files.setdefault(candidate.name, candidate)
        for file_name in referenced_file_names:
            source_path = available_files.get(file_name) or available_files.get(Path(file_name).name)
            if source_path is None:
                continue
            destination = target_dir / file_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                shutil.copy2(source_path, destination)
    return True


def _convert_annotation_to_item(
    *,
    annotation: dict[str, object],
    split: str,
    image_id: int,
    item_index: int,
    categories_by_id: dict[int, dict[str, object]],
    attributes_by_id: dict[int, dict[str, object]],
    category_map: dict[str, str],
    attribute_map: dict[str, dict[str, str]],
) -> tuple[V3DocumentItem, list[int], list[str]] | None:
    category = categories_by_id.get(int(annotation.get("category_id", 0)))
    if not category:
        return None
    normalized_category = _normalize_label(str(category.get("name", "")))
    mapped_category = category_map.get(normalized_category)
    if not mapped_category:
        return None

    colors: list[str] = []
    materials: list[str] = []
    patterns: list[str] = []
    textures: list[str] = []
    style_tags: list[str] = []
    unmapped_attribute_ids: list[int] = []
    unmapped_attribute_names: list[str] = []
    for attribute_id in annotation.get("attribute_ids", []):
        attribute_id_int = int(attribute_id)
        attribute = attributes_by_id.get(attribute_id_int)
        if not attribute:
            unmapped_attribute_ids.append(attribute_id_int)
            continue
        attribute_name = str(attribute.get("name", ""))
        mapped = attribute_map.get(_normalize_label(attribute_name))
        if not mapped:
            unmapped_attribute_ids.append(attribute_id_int)
            unmapped_attribute_names.append(attribute_name)
            continue
        field = mapped.get("field", "drop")
        value = mapped.get("value", "")
        if field == "drop" or not value:
            continue
        if field == "color":
            _append_unique(colors, value)
        elif field == "material":
            _append_unique(materials, value)
        elif field == "pattern":
            _append_unique(patterns, value)
        elif field == "texture":
            _append_unique(textures, value)
        elif field == "style_tags":
            _append_unique(style_tags, value)

    item = V3DocumentItem(
        item_id=f"fashionpedia:{split}:{image_id}#{item_index}",
        category=mapped_category,
        color=colors[:2],
        silhouette=[],
        material=materials[:2],
        pattern=patterns[:2],
        texture=textures[:2],
        style_tags=style_tags[:4],
        style_concepts=[],
        confidence=1.0,
        evidence=[
            f"fashionpedia_category:{str(category.get('name', '')).strip()}",
            *[
                f"fashionpedia_attribute:{attributes_by_id[attr_id]['name']}"
                for attr_id in annotation.get("attribute_ids", [])
                if int(attr_id) in attributes_by_id
            ],
        ],
        source="fashionpedia_import",
    )
    return item, unmapped_attribute_ids, unmapped_attribute_names


def _build_document_detail(items: list[V3DocumentItem]) -> str:
    return "|".join(_build_item_phrase(item) for item in items)


def _build_item_phrase(item: V3DocumentItem) -> str:
    tokens: list[str] = []
    if item.color:
        tokens.append(item.color[0])
    if item.material:
        tokens.append(item.material[0])
    if item.pattern:
        tokens.append(item.pattern[0])
    elif item.texture:
        tokens.append(item.texture[0])
    elif item.style_tags:
        tokens.append(item.style_tags[0])
    tokens.append(item.category)
    return " ".join(token for token in tokens if token).strip()


def _build_document_tags(items: list[V3DocumentItem], *, detail: str) -> dict[str, str]:
    values_by_field: dict[str, list[str]] = {field: [] for field in FEATURE_FIELDS}
    for item in items:
        _extend_unique(values_by_field["category"], [item.category])
        _extend_unique(values_by_field["color"], item.color)
        _extend_unique(values_by_field["material"], item.material)
        _extend_unique(values_by_field["pattern"], item.pattern)
        _extend_unique(values_by_field["texture"], item.texture)
    values_by_field["detail"] = [phrase for phrase in detail.split("|") if phrase]
    return {field: "|".join(values_by_field[field]) for field in FEATURE_FIELDS if values_by_field[field]}


def _annotation_sort_key(annotation: dict[str, object]) -> tuple[float, float, int]:
    bbox = annotation.get("bbox") or [0, 0, 0, 0]
    if not isinstance(bbox, list) or len(bbox) < 2:
        return (0.0, 0.0, int(annotation.get("id", 0)))
    return (float(bbox[1]), float(bbox[0]), int(annotation.get("id", 0)))


def _normalize_label(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return normalize_token(cleaned)


def _append_unique(values: list[str], value: str) -> None:
    normalized = normalize_token(value)
    if normalized and normalized not in values:
        values.append(normalized)


def _extend_unique(values: list[str], candidates: Iterable[str]) -> None:
    for candidate in candidates:
        _append_unique(values, candidate)


def _write_jsonl(path: str, records: list[dict[str, object]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
