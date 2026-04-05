from __future__ import annotations

import io
from urllib import error

import pytest

from switch_query.tagging.openai_batch_tagger import OpenAIBatchTransport


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return self.payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_request_bytes_retries_transient_http_errors(monkeypatch: pytest.MonkeyPatch) -> None:
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
        return FakeResponse(b'{"id":"ok"}')

    monkeypatch.setattr("switch_query.tagging.openai_batch_tagger.request.urlopen", fake_urlopen)
    monkeypatch.setattr("switch_query.tagging.openai_batch_tagger.time.sleep", lambda _: None)
    transport = OpenAIBatchTransport(
        api_key="test-key",
        max_retries=3,
        retry_backoff_seconds=0.01,
    )

    payload = transport._request_json(
        "https://api.openai.com/v1/files",
        method="GET",
        body=None,
        headers={"Authorization": "Bearer test-key"},
    )

    assert payload["id"] == "ok"
    assert attempts["count"] == 3

