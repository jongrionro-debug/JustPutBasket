from __future__ import annotations

import json
from pathlib import Path

from switch_query.v3.fashionpedia import convert_fashionpedia_to_v3_documents
from switch_query.v3.index import build_archive_index
from switch_query.v3.preprocessing import read_archive_documents_jsonl
from switch_query.v3.storage import JsonArchiveIndexStore


class FakeEncoder:
    def encode_image(self, image_paths):
        return [[float(index), 1.0] for index, _path in enumerate(image_paths, start=1)]


def _write_fashionpedia_fixture(root: Path) -> None:
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    (root / "images" / "train" / "look1.jpg").write_bytes(b"fake-jpg")

    train_payload = {
        "licenses": [{"id": 1, "name": "cc", "url": "https://license.test/cc"}],
        "images": [
            {
                "id": 1,
                "file_name": "look1.jpg",
                "license": 1,
                "original_url": "https://example.com/look1.jpg",
            }
        ],
        "categories": [
            {"id": 1, "name": "Dress"},
            {"id": 2, "name": "Shoe"},
            {"id": 3, "name": "Sleeve"},
        ],
        "attributes": [
            {"id": 1, "name": "Red"},
            {"id": 2, "name": "Leather"},
            {"id": 3, "name": "Striped"},
            {"id": 4, "name": "Mystery Trim"},
        ],
        "annotations": [
            {
                "id": 12,
                "image_id": 1,
                "category_id": 2,
                "attribute_ids": [1, 2],
                "bbox": [10, 260, 30, 30],
            },
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "attribute_ids": [1, 3, 4],
                "bbox": [20, 20, 120, 220],
            },
            {
                "id": 13,
                "image_id": 1,
                "category_id": 3,
                "attribute_ids": [],
                "bbox": [30, 30, 40, 40],
            },
        ],
    }
    val_payload = {
        "licenses": [{"id": 2, "name": "cc2", "url": "https://license.test/cc2"}],
        "images": [
            {
                "id": 2,
                "file_name": "look2.jpg",
                "license": 2,
                "original_url": "https://example.com/look2.jpg",
            }
        ],
        "categories": [{"id": 1, "name": "Dress"}],
        "attributes": [{"id": 1, "name": "Red"}],
        "annotations": [
            {
                "id": 21,
                "image_id": 2,
                "category_id": 1,
                "attribute_ids": [1],
                "bbox": [0, 0, 10, 10],
            }
        ],
    }
    (root / "annotations" / "instances_attributes_train2020.json").write_text(
        json.dumps(train_payload),
        encoding="utf-8",
    )
    (root / "annotations" / "instances_attributes_val2020.json").write_text(
        json.dumps(val_payload),
        encoding="utf-8",
    )


def test_convert_fashionpedia_to_v3_documents_writes_expected_documents_and_manifest(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "fashionpedia"
    documents_path = tmp_path / "item_enriched_documents_full.jsonl"
    provenance_path = tmp_path / "provenance_manifest.jsonl"
    _write_fashionpedia_fixture(dataset_root)

    result = convert_fashionpedia_to_v3_documents(
        dataset_root=str(dataset_root),
        output_path=str(documents_path),
        provenance_path=str(provenance_path),
    )

    documents = read_archive_documents_jsonl(str(documents_path))
    manifest_records = [
        json.loads(line) for line in provenance_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.document_count == 1
    assert result.converted_image_count == 1
    assert result.skipped_missing_image_count == 1
    assert len(documents) == 1
    assert documents[0].image_id == "fashionpedia:train:1"
    assert [item.item_id for item in documents[0].items] == [
        "fashionpedia:train:1#1",
        "fashionpedia:train:1#2",
    ]
    assert [item.category for item in documents[0].items] == ["dress", "shoes"]
    assert documents[0].items[0].color == ["red"]
    assert documents[0].items[0].pattern == ["striped"]
    assert documents[0].items[1].material == ["leather"]
    assert documents[0].detail == "red striped dress|red leather shoes"
    assert documents[0].canonical_tags["category"] == "dress|shoes"
    assert documents[0].raw_tags["detail"] == "red striped dress|red leather shoes"
    assert documents[0].item_extraction_notes == ["fashionpedia_import", "fashionpedia_split:train"]

    converted_manifest = manifest_records[0]
    missing_manifest = manifest_records[1]
    assert converted_manifest["status"] == "converted"
    assert converted_manifest["dropped_annotation_count"] == 1
    assert converted_manifest["unmapped_category_count"] == 1
    assert converted_manifest["unmapped_attribute_names"] == ["Mystery Trim"]
    assert missing_manifest["status"] == "missing_image"


def test_convert_fashionpedia_output_builds_v3_index(tmp_path: Path) -> None:
    dataset_root = tmp_path / "fashionpedia"
    documents_path = tmp_path / "item_enriched_documents_full.jsonl"
    provenance_path = tmp_path / "provenance_manifest.jsonl"
    index_path = tmp_path / "archive_index.json"
    _write_fashionpedia_fixture(dataset_root)

    convert_fashionpedia_to_v3_documents(
        dataset_root=str(dataset_root),
        output_path=str(documents_path),
        provenance_path=str(provenance_path),
    )
    documents = read_archive_documents_jsonl(str(documents_path))

    index = build_archive_index(documents, FakeEncoder(), JsonArchiveIndexStore(str(index_path)))

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    assert len(index.documents) == 1
    assert payload["documents"][0]["image_id"] == "fashionpedia:train:1"
    assert payload["feature_vocabulary"]["category"]["dress"] == "dress"
