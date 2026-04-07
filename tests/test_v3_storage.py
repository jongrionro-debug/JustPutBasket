from __future__ import annotations

from pathlib import Path

from switch_query.v3.models import V3ArchiveIndex, V3DocumentItem, V3IndexedDocument
from switch_query.v3.storage import InMemoryArchiveIndex, JsonArchiveIndexStore


def test_v3_in_memory_archive_index_round_trips_nested_items() -> None:
    store = InMemoryArchiveIndex()
    index = V3ArchiveIndex(
        documents=[
            V3IndexedDocument(
                image_id="look-1",
                file_path="/tmp/look-1.jpg",
                brand="alpha",
                season_group="spring-ready-to-wear",
                canonical_tags={"category": "coat", "color": "black"},
                raw_tags={"mood": "minimal"},
                detail="minimal black coat",
                items=[
                    V3DocumentItem(
                        item_id="look-1#1",
                        category="coat",
                        color=["black"],
                        style_tags=["minimal"],
                        confidence=0.9,
                        evidence=["detail:minimal black coat"],
                        source="test",
                    )
                ],
                item_confidence=0.9,
                item_extraction_notes=["single item"],
                vector=[0.1, 0.2, 0.3],
            )
        ],
        feature_vocabulary={"color": {"jet black": "black"}},
    )

    store.save(index)
    loaded = store.load()

    assert loaded.documents[0].items[0].category == "coat"
    assert loaded.documents[0].vector == [0.1, 0.2, 0.3]
    assert loaded.feature_vocabulary["color"]["jet black"] == "black"


def test_v3_json_archive_index_store_round_trips_nested_items(tmp_path: Path) -> None:
    store = JsonArchiveIndexStore(str(tmp_path / "archive_index.json"))
    index = V3ArchiveIndex(
        documents=[
            V3IndexedDocument(
                image_id="look-1",
                file_path="/tmp/look-1.jpg",
                brand="alpha",
                season_group="spring-ready-to-wear",
                canonical_tags={"category": "coat", "color": "black"},
                raw_tags={"mood": "minimal"},
                detail="minimal black coat",
                items=[
                    V3DocumentItem(
                        item_id="look-1#1",
                        category="coat",
                        color=["black"],
                        style_tags=["minimal"],
                        confidence=0.9,
                        evidence=["detail:minimal black coat"],
                        source="test",
                    )
                ],
                item_confidence=0.9,
                item_extraction_notes=["single item"],
                vector=[0.1, 0.2, 0.3],
            )
        ],
        feature_vocabulary={"color": {"jet black": "black"}},
    )

    store.save(index)
    loaded = store.load()

    assert loaded.documents[0].items[0].style_tags == ["minimal"]
    assert loaded.documents[0].vector == [0.1, 0.2, 0.3]
    assert loaded.feature_vocabulary["color"]["jet black"] == "black"
