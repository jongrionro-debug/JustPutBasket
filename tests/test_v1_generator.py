from __future__ import annotations

import os
from pathlib import Path

import pytest

from switch_query.v1.generator import LuxiaImageGenerator, LuxiaImageGeneratorConfig


class FakeResponse:
    def __init__(self, *, status_code=200, json_payload=None, text="", content=b"", headers=None):
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text
        self.content = content
        self.headers = headers or {}

    def json(self):
        if isinstance(self._json_payload, Exception):
            raise self._json_payload
        return self._json_payload


class FakeSession:
    def __init__(self, post_response: FakeResponse, get_responses: list[FakeResponse] | None = None):
        self.post_response = post_response
        self.get_responses = list(get_responses or [])
        self.post_calls = []
        self.get_calls = []

    def post(self, url, headers, json, timeout):
        self.post_calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        return self.post_response

    def get(self, url, timeout):
        self.get_calls.append({"url": url, "timeout": timeout})
        if not self.get_responses:
            raise AssertionError("No fake GET response configured")
        return self.get_responses.pop(0)


def test_luxia_generator_posts_expected_payload_and_downloads_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        post_response=FakeResponse(
            json_payload={
                "data": [
                    {"url": "https://cdn.example.com/generated-1.png", "revised_prompt": "rev 1"},
                    {"url": "https://cdn.example.com/generated-2.png", "revised_prompt": "rev 2"},
                ]
            }
        ),
        get_responses=[
            FakeResponse(content=b"img-1", headers={"Content-Type": "image/png"}),
            FakeResponse(content=b"img-2", headers={"Content-Type": "image/png"}),
        ],
    )
    generator = LuxiaImageGenerator(
        LuxiaImageGeneratorConfig(output_dir=str(tmp_path)),
        session=session,
    )

    paths = generator.generate(
        "black tailored coat",
        2,
        query_id="q1",
        balance_score=0.3,
    )

    assert len(paths) == 2
    assert Path(paths[0]).name == "q1_gen_01.png"
    assert Path(paths[1]).name == "q1_gen_02.png"
    assert Path(paths[0]).read_bytes() == b"img-1"
    assert session.post_calls[0]["url"].endswith("/image/openai/dalle3/hd/generate")
    assert session.post_calls[0]["headers"]["apikey"] == "secret-key"
    assert session.post_calls[0]["json"]["model"] == "dall-e-3"
    assert session.post_calls[0]["json"]["response_format"] == "url"
    assert "tightly aligned fashion reference" in session.post_calls[0]["json"]["prompt"]
    assert generator.generated_metadata[0].revised_prompt == "rev 1"


def test_luxia_generator_uses_divergent_prompt_for_negative_balance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        post_response=FakeResponse(json_payload={"url": "https://cdn.example.com/generated.png"}),
        get_responses=[FakeResponse(content=b"img", headers={"Content-Type": "image/png"})],
    )
    generator = LuxiaImageGenerator(
        LuxiaImageGeneratorConfig(output_dir=str(tmp_path)),
        session=session,
    )

    generator.generate("editorial black coat", 1, query_id="q2", balance_score=-0.5)

    assert "Produce diverse fashion directions" in session.post_calls[0]["json"]["prompt"]


def test_luxia_generator_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("LUXIA_API_KEY", raising=False)
    generator = LuxiaImageGenerator(LuxiaImageGeneratorConfig(output_dir=str(tmp_path)))

    with pytest.raises(RuntimeError, match="LUXIA_API_KEY is required"):
        generator.generate("black coat", 1, query_id="q1", balance_score=0.0)


def test_luxia_generator_rejects_http_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    generator = LuxiaImageGenerator(
        LuxiaImageGeneratorConfig(output_dir=str(tmp_path)),
        session=FakeSession(
            post_response=FakeResponse(status_code=500, text="server error"),
        ),
    )

    with pytest.raises(RuntimeError, match="status 500"):
        generator.generate("black coat", 1, query_id="q1", balance_score=0.0)


def test_luxia_generator_rejects_missing_url_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    generator = LuxiaImageGenerator(
        LuxiaImageGeneratorConfig(output_dir=str(tmp_path)),
        session=FakeSession(
            post_response=FakeResponse(json_payload={"data": [{"revised_prompt": "rev only"}]}),
        ),
    )

    with pytest.raises(RuntimeError, match="image urls"):
        generator.generate("black coat", 1, query_id="q1", balance_score=0.0)


def test_luxia_generator_accepts_top_level_list_payload(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    generator = LuxiaImageGenerator(
        LuxiaImageGeneratorConfig(output_dir=str(tmp_path)),
        session=FakeSession(
            post_response=FakeResponse(
                json_payload=[
                    {"url": "https://cdn.example.com/generated.png", "revised_prompt": "list payload"}
                ]
            ),
            get_responses=[FakeResponse(content=b"img", headers={"Content-Type": "image/png"})],
        ),
    )

    paths = generator.generate("black coat", 1, query_id="q1", balance_score=0.0)

    assert len(paths) == 1
    assert generator.generated_metadata[0].revised_prompt == "list payload"


def test_luxia_generator_rejects_download_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    generator = LuxiaImageGenerator(
        LuxiaImageGeneratorConfig(output_dir=str(tmp_path)),
        session=FakeSession(
            post_response=FakeResponse(json_payload={"url": "https://cdn.example.com/generated.png"}),
            get_responses=[FakeResponse(status_code=404)],
        ),
    )

    with pytest.raises(RuntimeError, match="Failed to download generated image"):
        generator.generate("black coat", 1, query_id="q1", balance_score=0.0)
