from __future__ import annotations

from dataclasses import asdict

from switch_query.v3.models import (
    V3ArchiveDocument,
    V3ArchiveIndex,
    V3DocumentItem,
    V3IndexedDocument,
    V3ItemExtractionInput,
    V3ItemExtractionOutput,
    V3ParsedQuery,
    V3TargetItem,
)
from switch_query.v3.item_extractor import parse_luxia_item_extraction_response


def test_v3_archive_document_supports_multi_item_schema() -> None:
    document = V3ArchiveDocument(
        image_id="2026:spring-ready-to-wear:test-brand:0001",
        file_path="/tmp/look-0001.jpg",
        brand="test-brand",
        season_group="spring-ready-to-wear",
        canonical_tags={
            "category": "jacket|trousers",
            "color": "black|white",
            "mood": "minimal",
        },
        raw_tags={
            "color": "black look with white trousers",
        },
        detail="black jacket|wide-leg white trousers",
        items=[
            V3DocumentItem(
                item_id="item_1",
                category="jacket",
                color=["black"],
                style_tags=["vintage"],
                confidence=0.86,
                evidence=["detail:black jacket"],
                source="detail_text",
            ),
            V3DocumentItem(
                item_id="item_2",
                category="trousers",
                color=["white"],
                silhouette=["wide-leg"],
                confidence=0.91,
                evidence=["detail:wide-leg white trousers"],
                source="detail_text",
            ),
        ],
        item_confidence=0.89,
        item_extraction_notes=["color split inferred from detail text"],
    )

    payload = asdict(document)

    assert payload["detail"] == "black jacket|wide-leg white trousers"
    assert payload["items"][0]["category"] == "jacket"
    assert payload["items"][0]["color"] == ["black"]
    assert payload["items"][1]["category"] == "trousers"
    assert payload["items"][1]["silhouette"] == ["wide-leg"]
    assert payload["item_confidence"] == 0.89
    assert payload["item_extraction_notes"] == ["color split inferred from detail text"]


def test_v3_archive_index_keeps_item_enriched_documents() -> None:
    index = V3ArchiveIndex(
        documents=[
            V3IndexedDocument(
                image_id="2026:spring-ready-to-wear:test-brand:0002",
                file_path="/tmp/look-0002.jpg",
                brand="test-brand",
                season_group="spring-ready-to-wear",
                canonical_tags={"category": "coat", "color": "black"},
                raw_tags={"mood": "minimal"},
                detail="minimal black coat",
                items=[
                    V3DocumentItem(
                        item_id="item_1",
                        category="coat",
                        color=["black"],
                        confidence=0.95,
                        evidence=["detail:minimal black coat"],
                        source="detail_text",
                    )
                ],
                item_confidence=0.95,
                item_extraction_notes=[],
                vector=[0.1, 0.2, 0.3],
            )
        ],
        feature_vocabulary={"color": {"jet black": "black"}},
    )

    assert len(index.documents) == 1
    assert index.documents[0].items[0].category == "coat"
    assert index.documents[0].vector == [0.1, 0.2, 0.3]
    assert index.feature_vocabulary["color"]["jet black"] == "black"


def test_v3_parsed_query_supports_multi_item_target_binding() -> None:
    parsed_query = V3ParsedQuery(
        query_text="white trousers with vintage black jacket",
        target_items=[
            V3TargetItem(
                target_item_id="item_1",
                category="trousers",
                color=["white"],
                required_attributes=["category", "color"],
                raw_phrase="white trousers",
            ),
            V3TargetItem(
                target_item_id="item_2",
                category="jacket",
                color=["black"],
                style_tags=["vintage"],
                required_attributes=["category", "color"],
                preferred_attributes=["style_tags"],
                raw_phrase="vintage black jacket",
            ),
        ],
        global_constraints={},
        style_preferences={},
        confidence=0.93,
    )

    payload = asdict(parsed_query)

    assert payload["query_text"] == "white trousers with vintage black jacket"
    assert payload["target_items"][0]["category"] == "trousers"
    assert payload["target_items"][0]["color"] == ["white"]
    assert payload["target_items"][1]["category"] == "jacket"
    assert payload["target_items"][1]["style_tags"] == ["vintage"]
    assert payload["target_items"][1]["preferred_attributes"] == ["style_tags"]
    assert payload["confidence"] == 0.93


def test_v3_item_extraction_input_and_output_capture_contract() -> None:
    extraction_input = V3ItemExtractionInput(
        image_id="2026:spring-ready-to-wear:test-brand:0003",
        file_path="/tmp/look-0003.jpg",
        brand="test-brand",
        season_group="spring-ready-to-wear",
        canonical_tags={"category": "jacket|trousers", "color": "black|white"},
        raw_tags={"detail": "black jacket|wide-leg white trousers"},
        detail="black jacket|wide-leg white trousers",
        image_path="/tmp/look-0003.jpg",
        extraction_mode="image_assisted",
    )
    extraction_output = V3ItemExtractionOutput(
        items=[
            V3DocumentItem(
                item_id="item_1",
                category="jacket",
                color=["black"],
                confidence=0.82,
                evidence=["detail:black jacket"],
                source="luxia_image_assisted",
            )
        ],
        item_confidence=0.82,
        item_extraction_notes=["single-item confidence is provisional"],
    )

    input_payload = asdict(extraction_input)
    output_payload = asdict(extraction_output)

    assert input_payload["image_id"].endswith(":0003")
    assert input_payload["extraction_mode"] == "image_assisted"
    assert input_payload["detail"] == "black jacket|wide-leg white trousers"
    assert output_payload["items"][0]["source"] == "luxia_image_assisted"
    assert output_payload["item_confidence"] == 0.82


def test_parse_luxia_item_extraction_response_coerces_structured_items() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"items":[{"item_id":"item_1","category":"jacket","color":["black"],'
                        '"silhouette":[],"material":[],"pattern":[],"texture":[],"style_tags":["vintage"],'
                        '"confidence":0.88,"evidence":["detail:black jacket"],"source":"luxia_image_assisted"}],'
                        '"item_confidence":0.88,"item_extraction_notes":["validated from image"]}'
                    )
                }
            }
        ]
    }

    output = parse_luxia_item_extraction_response(
        response,
        image_id="2026:spring-ready-to-wear:test-brand:0009",
        extraction_mode="image_assisted",
    )

    assert len(output.items) == 1
    assert output.items[0].item_id == "2026:spring-ready-to-wear:test-brand:0009#1"
    assert output.items[0].category == "jacket"
    assert output.items[0].style_tags == []
    assert output.items[0].style_concepts == ["vintage"]
    assert output.item_confidence == 0.88
    assert output.item_extraction_notes == ["validated from image"]
