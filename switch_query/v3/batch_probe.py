"""Helpers for probing Luxia OpenAI-style batch endpoint support."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib import error, request

LUXIA_OPENAI_BASE_URL = "https://bridge.luxiacloud.com/llm/openai"
PROMISING_HTTP_STATUSES = frozenset({200, 201, 202, 204, 400, 401, 403, 405, 409, 415, 422})


@dataclass(slots=True)
class LuxiaBatchProbeSpec:
    name: str
    method: str
    url: str
    body: bytes | None = None
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LuxiaBatchProbeResult:
    name: str
    method: str
    url: str
    classification: str
    http_status: int | None
    detail: str


class LuxiaBatchProbeTransport(Protocol):
    def request(
        self,
        url: str,
        *,
        method: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> tuple[int | None, str]:
        """Return a response status code and response body snippet."""

    def upload_file(
        self,
        url: str,
        *,
        api_key: str,
        purpose: str,
        file_name: str,
        file_bytes: bytes,
    ) -> tuple[int | None, str]:
        """Upload a multipart file payload and return status plus text."""


@dataclass(slots=True)
class StdlibLuxiaBatchProbeTransport:
    timeout_seconds: float = 60.0

    def request(
        self,
        url: str,
        *,
        method: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> tuple[int | None, str]:
        http_request = request.Request(url=url, method=method, data=body, headers=headers)
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                payload = response.read().decode("utf-8", errors="replace")
                return response.status, payload
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            return exc.code, detail
        except error.URLError as exc:
            return None, str(exc)

    def upload_file(
        self,
        url: str,
        *,
        api_key: str,
        purpose: str,
        file_name: str,
        file_bytes: bytes,
    ) -> tuple[int | None, str]:
        boundary = f"----CodexBoundary{uuid.uuid4().hex}"
        body = self._build_multipart_body(
            boundary=boundary,
            purpose=purpose,
            file_name=file_name,
            file_bytes=file_bytes,
        )
        headers = {
            "apikey": api_key,
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        return self.request(url, method="POST", body=body, headers=headers)

    def _build_multipart_body(
        self,
        *,
        boundary: str,
        purpose: str,
        file_name: str,
        file_bytes: bytes,
    ) -> bytes:
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            purpose.encode("utf-8"),
            b"\r\n",
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'.encode("utf-8"),
            b"Content-Type: application/jsonl\r\n\r\n",
            file_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        return b"".join(parts)


@dataclass(slots=True)
class LuxiaBatchSubmitProbeResult:
    step: str
    endpoint_name: str
    url: str
    classification: str
    http_status: int | None
    detail: str
    file_id: str = ""
    batch_id: str = ""


def build_luxia_batch_probe_specs(*, api_key: str = "") -> list[LuxiaBatchProbeSpec]:
    common_headers: dict[str, str] = {}
    if api_key:
        common_headers["apikey"] = api_key

    batch_create_body = json.dumps(
        {
            "input_file_id": "probe-file-id",
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        }
    ).encode("utf-8")

    return [
        LuxiaBatchProbeSpec(
            name="files_list",
            method="GET",
            url=f"{LUXIA_OPENAI_BASE_URL}/files",
            headers=dict(common_headers),
        ),
        LuxiaBatchProbeSpec(
            name="files_create",
            method="POST",
            url=f"{LUXIA_OPENAI_BASE_URL}/files/create",
            body=b"{}",
            headers={
                **common_headers,
                "Content-Type": "application/json",
            },
        ),
        LuxiaBatchProbeSpec(
            name="batches_list",
            method="GET",
            url=f"{LUXIA_OPENAI_BASE_URL}/batches",
            headers=dict(common_headers),
        ),
        LuxiaBatchProbeSpec(
            name="batches_create",
            method="POST",
            url=f"{LUXIA_OPENAI_BASE_URL}/batches/create",
            body=batch_create_body,
            headers={
                **common_headers,
                "Content-Type": "application/json",
            },
        ),
        LuxiaBatchProbeSpec(
            name="batches_plain_post",
            method="POST",
            url=f"{LUXIA_OPENAI_BASE_URL}/batches",
            body=batch_create_body,
            headers={
                **common_headers,
                "Content-Type": "application/json",
            },
        ),
    ]


def run_luxia_batch_capability_probe(
    *,
    api_key_env: str = "LUXIA_API_KEY",
    transport: LuxiaBatchProbeTransport | None = None,
) -> list[LuxiaBatchProbeResult]:
    if transport is None:
        transport = StdlibLuxiaBatchProbeTransport()

    api_key = os.environ.get(api_key_env, "")
    results: list[LuxiaBatchProbeResult] = []
    for spec in build_luxia_batch_probe_specs(api_key=api_key):
        http_status, detail = transport.request(
            spec.url,
            method=spec.method,
            body=spec.body,
            headers=spec.headers,
        )
        results.append(
            LuxiaBatchProbeResult(
                name=spec.name,
                method=spec.method,
                url=spec.url,
                classification=_classify_probe_response(http_status),
                http_status=http_status,
                detail=_truncate_detail(detail),
            )
        )
    return results


def has_promising_batch_probe_result(results: list[LuxiaBatchProbeResult]) -> bool:
    return any(result.classification == "promising" for result in results)


def build_luxia_batch_submit_probe_jsonl(*, model: str, endpoint_path: str) -> bytes:
    line = {
        "custom_id": "luxia-batch-probe-1",
        "method": "POST",
        "url": endpoint_path,
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only compact JSON.",
                },
                {
                    "role": "user",
                    "content": "Return {\"ok\":true,\"mode\":\"batch_probe\"}",
                },
            ],
            "stream": False,
        },
    }
    return (json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8")


def run_luxia_batch_submit_probe(
    *,
    api_key_env: str = "LUXIA_API_KEY",
    model: str = "gpt-4o-2024-08-06",
    endpoint_path: str = "/v1/chat/completions",
    transport: LuxiaBatchProbeTransport | None = None,
) -> list[LuxiaBatchSubmitProbeResult]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"{api_key_env} is required for submit-batch-probe")
    if transport is None:
        transport = StdlibLuxiaBatchProbeTransport()

    probe_file_bytes = build_luxia_batch_submit_probe_jsonl(model=model, endpoint_path=endpoint_path)
    results: list[LuxiaBatchSubmitProbeResult] = []

    upload_attempts = [
        ("files", f"{LUXIA_OPENAI_BASE_URL}/files"),
        ("files_create", f"{LUXIA_OPENAI_BASE_URL}/files/create"),
    ]
    file_id = ""
    for endpoint_name, url in upload_attempts:
        http_status, detail = transport.upload_file(
            url,
            api_key=api_key,
            purpose="batch",
            file_name="luxia_batch_probe.jsonl",
            file_bytes=probe_file_bytes,
        )
        payload = _load_json_if_possible(detail)
        file_id = _extract_id(payload)
        results.append(
            LuxiaBatchSubmitProbeResult(
                step="upload",
                endpoint_name=endpoint_name,
                url=url,
                classification=_classify_submit_step(http_status, success=bool(file_id)),
                http_status=http_status,
                detail=_truncate_detail(detail),
                file_id=file_id,
            )
        )
        if file_id:
            break

    if not file_id:
        return results

    batch_body = json.dumps(
        {
            "input_file_id": file_id,
            "endpoint": endpoint_path,
            "completion_window": "24h",
        }
    ).encode("utf-8")
    batch_attempts = [
        ("batches_create", f"{LUXIA_OPENAI_BASE_URL}/batches/create"),
        ("batches", f"{LUXIA_OPENAI_BASE_URL}/batches"),
    ]
    headers = {
        "apikey": api_key,
        "Content-Type": "application/json",
    }
    for endpoint_name, url in batch_attempts:
        http_status, detail = transport.request(
            url,
            method="POST",
            body=batch_body,
            headers=headers,
        )
        payload = _load_json_if_possible(detail)
        batch_id = _extract_id(payload)
        results.append(
            LuxiaBatchSubmitProbeResult(
                step="create_batch",
                endpoint_name=endpoint_name,
                url=url,
                classification=_classify_submit_step(http_status, success=bool(batch_id)),
                http_status=http_status,
                detail=_truncate_detail(detail),
                file_id=file_id,
                batch_id=batch_id,
            )
        )
        if batch_id:
            break
    return results


def _classify_probe_response(http_status: int | None) -> str:
    if http_status is None:
        return "network_error"
    if http_status == 404:
        return "missing"
    if http_status in PROMISING_HTTP_STATUSES:
        return "promising"
    return "unexpected"


def _classify_submit_step(http_status: int | None, *, success: bool) -> str:
    if success:
        return "success"
    return _classify_probe_response(http_status)


def _load_json_if_possible(detail: str) -> dict[str, object] | None:
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _extract_id(payload: dict[str, object] | None) -> str:
    if not payload:
        return ""
    value = payload.get("id")
    if value is None:
        return ""
    return str(value).strip()


def _truncate_detail(detail: str, *, limit: int = 200) -> str:
    cleaned = " ".join(detail.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."
