from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from switch_query.v3.item_extractor import (
    LuxiaItemExtractor,
    LuxiaItemExtractorConfig,
    build_luxia_item_extraction_request,
    parse_luxia_item_extraction_response,
)
from switch_query.v3.models import V3ItemExtractionInput


class FakeTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object], str]] = []

    def post_json(self, url: str, payload: dict[str, object], *, api_key: str) -> dict[str, object]:
        self.calls.append((url, payload, api_key))
        return self.response


def build_input(tmp_path: Path, *, suffix: str = ".jpg", extraction_mode: str = "text_only") -> V3ItemExtractionInput:
    image_path = tmp_path / f"look{suffix}"
    image_path.write_bytes(b"fake-image")
    return V3ItemExtractionInput(
        image_id="2026:spring-ready-to-wear:test-brand:0001",
        file_path=str(image_path),
        brand="test-brand",
        season_group="spring-ready-to-wear",
        canonical_tags={"category": "jacket|trousers", "color": "black|white"},
        raw_tags={"detail": "black jacket|wide-leg white trousers"},
        detail="black jacket|wide-leg white trousers",
        image_path=str(image_path),
        extraction_mode=extraction_mode,  # type: ignore[arg-type]
    )


def test_build_luxia_item_extraction_request_keeps_text_only_as_text(tmp_path: Path) -> None:
    extraction_input = build_input(tmp_path, extraction_mode="text_only")

    payload = build_luxia_item_extraction_request(extraction_input, model="gpt-4o-2024-08-06")

    assert payload["model"] == "gpt-4o-2024-08-06"
    assert payload["stream"] is False
    user_content = payload["messages"][1]["content"]
    assert len(user_content) == 1
    assert user_content[0]["type"] == "text"
    prompt_text = user_content[0]["text"]
    assert "Never put vintage, minimal, romantic, avant-garde, retro, edgy, elegant, modern, sporty into style_tags." in prompt_text
    assert '"items":[{"category":"","confidence":0.0}]' in prompt_text


def test_build_luxia_item_extraction_request_adds_data_url_image(tmp_path: Path) -> None:
    extraction_input = build_input(tmp_path, extraction_mode="image_assisted")

    payload = build_luxia_item_extraction_request(extraction_input, model="gpt-4o-2024-08-06")

    user_content = payload["messages"][1]["content"]
    assert len(user_content) == 2
    image_part = user_content[1]
    assert image_part["type"] == "image_url"
    image_url = image_part["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_url.split(",", 1)[1]) == b"fake-image"


def test_build_luxia_item_extraction_request_safe_resize_downsizes_large_images(tmp_path: Path) -> None:
    pillow = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "large-look.jpg"
    image = pillow.new("RGB", (2400, 1600), color=(20, 20, 20))
    image.save(image_path, format="JPEG", quality=95)
    extraction_input = V3ItemExtractionInput(
        image_id="2026:spring-ready-to-wear:test-brand:resize",
        file_path=str(image_path),
        brand="test-brand",
        season_group="spring-ready-to-wear",
        detail="large image resize probe",
        image_path=str(image_path),
        extraction_mode="image_assisted",
    )

    payload = build_luxia_item_extraction_request(
        extraction_input,
        model="gpt-4o-2024-08-06",
        image_transfer_mode="safe_resize",
        max_image_edge=1024,
        jpeg_quality=85,
    )

    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    resized_bytes = base64.b64decode(image_url.split(",", 1)[1])
    resized = pillow.open(io.BytesIO(resized_bytes))
    assert max(resized.size) <= 1024
    assert len(resized_bytes) < image_path.stat().st_size


@pytest.mark.parametrize(
    ("suffix", "expected_prefix"),
    [
        (".png", "data:image/png;base64,"),
        (".webp", "data:image/webp;base64,"),
    ],
)
def test_build_luxia_item_extraction_request_uses_correct_mime_type(
    tmp_path: Path,
    suffix: str,
    expected_prefix: str,
) -> None:
    extraction_input = build_input(tmp_path, suffix=suffix, extraction_mode="image_assisted")

    payload = build_luxia_item_extraction_request(extraction_input, model="gpt-4o-2024-08-06")

    image_url = payload["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith(expected_prefix)


def test_parse_luxia_item_extraction_response_reads_openai_style_content() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": """```json
{
  "items": [
    {
      "item_id": "ignored",
      "category": "jacket",
      "color": ["black"],
      "silhouette": [],
      "material": [],
      "pattern": [],
      "texture": [],
      "style_tags": ["vintage"],
      "confidence": 0.88,
      "evidence": ["detail:black jacket"],
      "source": "luxia_image_assisted"
    }
  ],
  "item_confidence": 0.88,
  "item_extraction_notes": ["validated from image"]
}
```"""
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
    assert output.items[0].style_tags == []
    assert output.items[0].style_concepts == ["vintage"]
    assert output.items[0].source == "luxia_image_assisted"
    assert output.item_confidence == 0.88


def test_parse_luxia_item_extraction_response_defaults_missing_compact_fields() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"items":[{"category":"dress","style_tags":["romantic"],"confidence":0.72}],"item_confidence":0.72}'
                }
            }
        ]
    }

    output = parse_luxia_item_extraction_response(
        response,
        image_id="2026:spring-ready-to-wear:test-brand:0010",
        extraction_mode="image_assisted",
    )

    assert len(output.items) == 1
    assert output.items[0].category == "dress"
    assert output.items[0].style_tags == []
    assert output.items[0].style_concepts == ["romantic"]
    assert output.items[0].evidence == []
    assert output.items[0].source == "luxia_image_assisted"
    assert output.item_extraction_notes == []


def test_parse_luxia_item_extraction_response_moves_concepts_out_of_style_tags() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"items":[{"category":"dress","style_tags":["avant-garde","cropped"],'
                        '"style_concepts":[],"confidence":0.72}],"item_confidence":0.72}'
                    )
                }
            }
        ]
    }

    output = parse_luxia_item_extraction_response(
        response,
        image_id="2026:spring-ready-to-wear:test-brand:concept-cleanup",
        extraction_mode="image_assisted",
    )

    assert output.items[0].style_tags == ["cropped"]
    assert output.items[0].style_concepts == ["avant garde"]


def test_parse_luxia_item_extraction_response_accepts_trailing_text_after_json() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": (
                        '{"items":[{"category":"dress","confidence":0.72}],"item_confidence":0.72}'
                        "\nNote: parsed successfully."
                    )
                }
            }
        ]
    }

    output = parse_luxia_item_extraction_response(
        response,
        image_id="2026:spring-ready-to-wear:test-brand:0011",
        extraction_mode="image_assisted",
    )

    assert len(output.items) == 1
    assert output.items[0].category == "dress"
    assert output.item_confidence == 0.72


def test_parse_luxia_item_extraction_response_requires_items_list() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": '{"item_confidence":0.5,"item_extraction_notes":[]}'
                }
            }
        ]
    }

    with pytest.raises(RuntimeError, match="items list"):
        parse_luxia_item_extraction_response(response)


def test_luxia_item_extractor_runs_end_to_end_with_fake_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extraction_input = build_input(tmp_path, extraction_mode="image_assisted")
    transport = FakeTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"items":[{"category":"jacket","color":["black"],"silhouette":[],"material":[],'
                            '"pattern":[],"texture":[],"style_tags":["vintage"],"confidence":0.87,'
                            '"evidence":["detail:black jacket"],"source":"luxia_image_assisted"}],'
                            '"item_confidence":0.87,"item_extraction_notes":["single item"]}'
                        )
                    }
                }
            ]
        }
    )
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    extractor = LuxiaItemExtractor(
        config=LuxiaItemExtractorConfig(model="gpt-4o-2024-08-06"),
        transport=transport,
    )

    output = extractor.extract_items(extraction_input)

    assert output.items[0].category == "jacket"
    assert output.items[0].source == "luxia_image_assisted"
    assert transport.calls[0][0].endswith("/gpt-4o/create")
    assert transport.calls[0][1]["stream"] is False
    assert transport.calls[0][2] == "secret-key"


def test_luxia_item_extractor_requires_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUXIA_API_KEY", raising=False)
    extractor = LuxiaItemExtractor(
        config=LuxiaItemExtractorConfig(),
        transport=FakeTransport({}),
    )

    with pytest.raises(RuntimeError, match="LUXIA_API_KEY is required"):
        extractor.extract_items(build_input(tmp_path, extraction_mode="text_only"))


def test_luxia_item_extractor_enriches_style_concepts_from_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_input = build_input(tmp_path, extraction_mode="text_only")
    extraction_input = V3ItemExtractionInput(
        image_id=base_input.image_id,
        file_path=base_input.file_path,
        brand=base_input.brand,
        season_group=base_input.season_group,
        canonical_tags={"category": "pants", "era": "vintage"},
        raw_tags={"mood": "vintage worn-in"},
        detail="vintage loose pants with a washed finish",
        image_path=base_input.image_path,
        extraction_mode=base_input.extraction_mode,
    )
    transport = FakeTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"items":[{"category":"pants","style_tags":[],"confidence":0.81}],'
                            '"item_confidence":0.81,"item_extraction_notes":["llm output"]}'
                        )
                    }
                }
            ]
        }
    )
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    extractor = LuxiaItemExtractor(
        config=LuxiaItemExtractorConfig(model="gpt-4o-2024-08-06"),
        transport=transport,
    )

    output = extractor.extract_items(extraction_input)

    assert output.items[0].style_concepts == ["vintage"]
    assert "items enriched from canonical/raw mood-era-detail context" in output.item_extraction_notes
