from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from urllib import error

import pytest

from switch_query.tagging.openai_batch_tagger import build_batch_input_jsonl
from switch_query.tagging.openai_vlm_tagger import (
    OpenAIHTTPTransport,
    OpenAIJsonTagger,
    build_openai_tag_request,
    build_raw_tag_row,
    parse_openai_tag_response,
    parse_openai_tag_response_text,
)
from switch_query.tagging.preprocessing import SampleRow, TaggingResult
from switch_query.v2.preprocessing import build_tagger


class FakeTransport:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        self.calls.append((url, payload))
        return self.response


def build_sample(tmp_path: Path, *, image_id: str = "look-1") -> SampleRow:
    image_path = tmp_path / f"{image_id}.jpg"
    image_path.write_bytes(b"fake-jpg")
    return SampleRow(
        image_id=image_id,
        file_path=str(image_path),
        season_group="spring-ready-to-wear",
        year="2026",
        brand="alpha",
        source_type="collection",
        filename=image_path.name,
        sample_reason="brand_first",
    )


def test_build_openai_tag_request_creates_stable_payload(tmp_path: Path) -> None:
    sample = build_sample(tmp_path)

    request_item = build_openai_tag_request(sample, model="gpt-4.1-mini")

    assert request_item.custom_id == sample.image_id
    assert request_item.payload["model"] == "gpt-4.1-mini"
    image_part = request_item.payload["input"][0]["content"][1]
    assert image_part["type"] == "input_image"
    assert image_part["detail"] == "low"
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")
    assert base64.b64decode(image_part["image_url"].split(",", 1)[1]) == b"fake-jpg"


def test_build_openai_tag_request_supports_file_id_image_inputs(tmp_path: Path) -> None:
    sample = build_sample(tmp_path)

    request_item = build_openai_tag_request(
        sample,
        model="gpt-4.1-mini",
        image_file_id="file-image-123",
    )

    image_part = request_item.payload["input"][0]["content"][1]
    assert image_part["type"] == "input_image"
    assert image_part["file_id"] == "file-image-123"
    assert image_part["detail"] == "low"


def test_openai_json_tagger_tags_image_and_builds_data_uri(tmp_path: Path) -> None:
    sample = build_sample(tmp_path)
    transport = FakeTransport(
        {
            "output_text": (
                '{"caption":"black wool coat","category":["coat","trousers"],'
                '"silhouette":"tailored","color":"black","material":"wool",'
                '"pattern":"solid","texture":"smooth","mood":"minimal",'
                '"season":"fall","era":"modern","detail":["long coat","trousers"],'
                '"review_needed":false,"confidence_note":"high"}'
            )
        }
    )
    tagger = OpenAIJsonTagger(model="gpt-4.1-mini", transport=transport)

    result = tagger.tag_image(sample)

    assert result.caption == "black wool coat"
    assert result.category == "coat|trousers"
    assert result.detail == "long coat|trousers"
    assert result.review_needed is False
    url, payload = transport.calls[0]
    assert url.endswith("/v1/responses")
    image_part = payload["input"][0]["content"][1]
    assert image_part["image_url"].startswith("data:image/jpeg;base64,")


def test_parse_openai_tag_response_extracts_text_from_output_blocks(tmp_path: Path) -> None:
    sample = build_sample(tmp_path, image_id="look-2")
    response = {
        "output": [
            {
                "content": [
                    {
                        "text": (
                            '{"caption":"red dress","category":"dress","color":"red",'
                            '"review_needed":true,"confidence_note":"uncertain"}'
                        )
                    }
                ]
            }
        ]
    }

    result = parse_openai_tag_response(sample=sample, response=response)

    assert result.caption == "red dress"
    assert result.color == "red"
    assert result.review_needed is True


def test_parse_openai_tag_response_text_marks_invalid_output() -> None:
    payload = parse_openai_tag_response_text("not json")

    assert payload["review_needed"] is True
    assert payload["confidence_note"] == "Model output was not valid JSON"


def test_build_raw_tag_row_converts_tagging_result_to_csv_shape(tmp_path: Path) -> None:
    sample = build_sample(tmp_path)
    raw_row = build_raw_tag_row(
        sample,
        TaggingResult(
            caption="black coat",
            category="coat",
            silhouette="tailored",
            color="black",
            material="wool",
            mood="minimal",
            review_needed=False,
            confidence_note="high",
        ),
    )

    assert raw_row.image_id == sample.image_id
    assert raw_row.raw_category == "coat"
    assert raw_row.raw_color == "black"
    assert raw_row.review_needed == "false"


def test_build_batch_input_jsonl_writes_deterministic_custom_ids(tmp_path: Path) -> None:
    samples = [build_sample(tmp_path, image_id="look-1"), build_sample(tmp_path, image_id="look-2")]
    output_path = tmp_path / "batch.jsonl"
    image_file_ids = {"look-1": "file-look-1", "look-2": "file-look-2"}

    row_count = build_batch_input_jsonl(
        samples,
        output_path=str(output_path),
        model="gpt-4.1-mini",
        image_file_ids=image_file_ids,
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert row_count == 2
    assert json.loads(lines[0])["custom_id"] == "look-1"
    assert json.loads(lines[0])["body"]["input"][0]["content"][1]["file_id"] == "file-look-1"
    assert json.loads(lines[1])["url"] == "/v1/responses"


def test_v2_build_tagger_supports_openai_aliases(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    tagger = build_tagger(
        tagger_type="openai-sync",
        model_name="mlx-community/Qwen2-VL-2B-Instruct-4bit",
        api_key_env="OPENAI_API_KEY",
        raw_output_log_dir="tmp/logs",
        tagger_command=None,
    )

    assert isinstance(tagger, OpenAIJsonTagger)
    assert tagger.model == "gpt-4.1-mini"
    assert tagger.api_key_env == "OPENAI_API_KEY"


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_openai_http_transport_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts = {"count": 0}

    def fake_urlopen(http_request, timeout):  # noqa: ANN001
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise error.HTTPError(
                url=http_request.full_url,
                code=520,
                msg="unknown",
                hdrs=None,
                fp=io.BytesIO(b"temporary"),
            )
        return FakeResponse(b'{"output_text":"{\\"caption\\":\\"ok\\"}"}')

    monkeypatch.setattr("switch_query.tagging.openai_vlm_tagger.request.urlopen", fake_urlopen)
    monkeypatch.setattr("switch_query.tagging.openai_vlm_tagger.time.sleep", lambda _: None)
    transport = OpenAIHTTPTransport(
        api_key="test-key",
        max_retries=3,
        retry_backoff_seconds=0.01,
    )

    payload = transport.post_json("https://api.openai.com/v1/responses", {"model": "gpt-4.1-mini"})

    assert payload["output_text"] == '{"caption":"ok"}'
    assert attempts["count"] == 3
