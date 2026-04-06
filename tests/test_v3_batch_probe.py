from __future__ import annotations

from switch_query.v3.batch_probe import (
    LUXIA_OPENAI_BASE_URL,
    LuxiaBatchProbeSpec,
    has_promising_batch_probe_result,
    build_luxia_batch_submit_probe_jsonl,
    run_luxia_batch_capability_probe,
    run_luxia_batch_submit_probe,
)


def test_run_luxia_batch_capability_probe_classifies_candidate_endpoints(monkeypatch) -> None:
    class FakeTransport:
        def request(
            self,
            url: str,
            *,
            method: str,
            body: bytes | None,
            headers: dict[str, str],
        ) -> tuple[int | None, str]:
            if url.endswith("/files"):
                assert method == "GET"
                return 401, '{"error":"missing auth"}'
            if url.endswith("/files/create"):
                assert method == "POST"
                assert body == b"{}"
                return 415, '{"error":"unsupported media type"}'
            if url.endswith("/batches") and method == "GET":
                return 404, '{"error":"not found"}'
            if url.endswith("/batches/create"):
                assert body is not None
                return 400, '{"error":"missing input_file_id"}'
            if url.endswith("/batches") and method == "POST":
                return None, "temporary DNS failure"
            raise AssertionError(f"Unexpected probe request: {method} {url}")

    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    results = run_luxia_batch_capability_probe(transport=FakeTransport())

    assert [result.name for result in results] == [
        "files_list",
        "files_create",
        "batches_list",
        "batches_create",
        "batches_plain_post",
    ]
    assert results[0].classification == "promising"
    assert results[0].http_status == 401
    assert results[1].classification == "promising"
    assert results[1].http_status == 415
    assert results[2].classification == "missing"
    assert results[2].http_status == 404
    assert results[3].classification == "promising"
    assert results[4].classification == "network_error"
    assert has_promising_batch_probe_result(results) is True


def test_probe_specs_include_expected_openai_style_paths(monkeypatch) -> None:
    class CaptureTransport:
        def __init__(self) -> None:
            self.requests: list[LuxiaBatchProbeSpec] = []

        def request(
            self,
            url: str,
            *,
            method: str,
            body: bytes | None,
            headers: dict[str, str],
        ) -> tuple[int | None, str]:
            self.requests.append(
                LuxiaBatchProbeSpec(
                    name=url.rsplit("/", 1)[-1],
                    method=method,
                    url=url,
                    body=body,
                    headers=headers,
                )
            )
            return 404, ""

    monkeypatch.delenv("LUXIA_API_KEY", raising=False)
    transport = CaptureTransport()

    run_luxia_batch_capability_probe(transport=transport)

    assert [item.url for item in transport.requests] == [
        f"{LUXIA_OPENAI_BASE_URL}/files",
        f"{LUXIA_OPENAI_BASE_URL}/files/create",
        f"{LUXIA_OPENAI_BASE_URL}/batches",
        f"{LUXIA_OPENAI_BASE_URL}/batches/create",
        f"{LUXIA_OPENAI_BASE_URL}/batches",
    ]
    assert all("apikey" not in item.headers for item in transport.requests)


def test_build_luxia_batch_submit_probe_jsonl_contains_single_chat_request() -> None:
    payload = build_luxia_batch_submit_probe_jsonl(
        model="gpt-4o-2024-08-06",
        endpoint_path="/v1/chat/completions",
    ).decode("utf-8")

    assert '"custom_id": "luxia-batch-probe-1"' in payload
    assert '"url": "/v1/chat/completions"' in payload
    assert '"model": "gpt-4o-2024-08-06"' in payload


def test_run_luxia_batch_submit_probe_uploads_and_creates_batch(monkeypatch) -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.upload_calls: list[str] = []
            self.request_calls: list[str] = []

        def upload_file(
            self,
            url: str,
            *,
            api_key: str,
            purpose: str,
            file_name: str,
            file_bytes: bytes,
        ) -> tuple[int | None, str]:
            self.upload_calls.append(url)
            assert api_key == "secret-key"
            assert purpose == "batch"
            assert file_name == "luxia_batch_probe.jsonl"
            assert file_bytes.endswith(b"\n")
            if url.endswith("/files"):
                return 404, '{"error":"not found"}'
            return 200, '{"id":"file-probe-1","object":"file"}'

        def request(
            self,
            url: str,
            *,
            method: str,
            body: bytes | None,
            headers: dict[str, str],
        ) -> tuple[int | None, str]:
            self.request_calls.append(url)
            assert method == "POST"
            assert headers["apikey"] == "secret-key"
            assert body is not None
            if url.endswith("/batches/create"):
                return 200, '{"id":"batch-probe-1","status":"validating"}'
            return 404, '{"error":"not found"}'

    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    transport = FakeTransport()

    results = run_luxia_batch_submit_probe(transport=transport)

    assert [result.step for result in results] == ["upload", "upload", "create_batch"]
    assert results[0].classification == "missing"
    assert results[1].classification == "success"
    assert results[1].file_id == "file-probe-1"
    assert results[2].classification == "success"
    assert results[2].batch_id == "batch-probe-1"
    assert transport.upload_calls == [
        f"{LUXIA_OPENAI_BASE_URL}/files",
        f"{LUXIA_OPENAI_BASE_URL}/files/create",
    ]
    assert transport.request_calls == [f"{LUXIA_OPENAI_BASE_URL}/batches/create"]
