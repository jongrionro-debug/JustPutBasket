"""Batch API helpers for OpenAI-backed archive tagging."""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib import error, request

from .openai_vlm_tagger import (
    DEFAULT_OPENAI_VISION_MODEL,
    OPENAI_BATCHES_URL,
    OPENAI_FILES_URL,
    build_openai_tag_request,
    build_raw_tag_row,
    parse_openai_tag_response,
)
from .preprocessing import RawTagRow, SampleRow


@dataclass(slots=True)
class OpenAIBatchJob:
    job_id: str
    input_file_id: str
    output_file_id: str | None
    error_file_id: str | None
    status: str


@dataclass(slots=True)
class UploadedImageFile:
    image_id: str
    file_path: str
    file_id: str


class OpenAIBatchTransport:
    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 300.0,
        max_retries: int = 5,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def upload_file(self, path: str, *, purpose: str = "batch") -> dict[str, object]:
        boundary = f"----CodexBoundary{uuid.uuid4().hex}"
        file_path = Path(path)
        body = self._build_multipart_body(boundary=boundary, purpose=purpose, file_path=file_path)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        }
        return self._request_json(OPENAI_FILES_URL, method="POST", body=body, headers=headers)

    def create_batch(self, input_file_id: str) -> dict[str, object]:
        body = json.dumps(
            {
                "input_file_id": input_file_id,
                "endpoint": "/v1/responses",
                "completion_window": "24h",
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        return self._request_json(OPENAI_BATCHES_URL, method="POST", body=body, headers=headers)

    def retrieve_batch(self, batch_id: str) -> dict[str, object]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return self._request_json(
            f"{OPENAI_BATCHES_URL}/{batch_id}",
            method="GET",
            body=None,
            headers=headers,
        )

    def download_file_text(self, file_id: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = self._request_bytes(
            f"{OPENAI_FILES_URL}/{file_id}/content",
            method="GET",
            body=None,
            headers=headers,
        )
        return payload.decode("utf-8")

    def _request_json(
        self,
        url: str,
        *,
        method: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> dict[str, object]:
        payload = self._request_bytes(url, method=method, body=body, headers=headers)
        return json.loads(payload.decode("utf-8"))

    def _request_bytes(
        self,
        url: str,
        *,
        method: str,
        body: bytes | None,
        headers: dict[str, str],
    ) -> bytes:
        retryable_http_codes = {429, 500, 502, 503, 504, 520}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            http_request = request.Request(url=url, method=method, data=body, headers=headers)
            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    return response.read()
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in retryable_http_codes and attempt < self.max_retries:
                    _sleep_before_retry(self.retry_backoff_seconds, attempt)
                    last_error = RuntimeError(f"OpenAI request failed: {exc.code} {detail}")
                    continue
                raise RuntimeError(f"OpenAI request failed: {exc.code} {detail}") from exc
            except (error.URLError, BrokenPipeError) as exc:
                if attempt < self.max_retries:
                    _sleep_before_retry(self.retry_backoff_seconds, attempt)
                    last_error = exc
                    continue
                raise RuntimeError(f"OpenAI request failed: {exc}") from exc
        if last_error is not None:
            raise RuntimeError(f"OpenAI request failed after retries: {last_error}") from last_error
        raise RuntimeError("OpenAI request failed without a captured exception")

    def _build_multipart_body(
        self,
        *,
        boundary: str,
        purpose: str,
        file_path: Path,
    ) -> bytes:
        mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        parts = [
            f"--{boundary}\r\n".encode("utf-8"),
            b'Content-Disposition: form-data; name="purpose"\r\n\r\n',
            purpose.encode("utf-8"),
            b"\r\n",
            f"--{boundary}\r\n".encode("utf-8"),
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'.encode("utf-8"),
            f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("utf-8"),
        ]
        return b"".join(parts)


def build_batch_input_jsonl(
    samples: Iterable[SampleRow],
    *,
    output_path: str,
    model: str = DEFAULT_OPENAI_VISION_MODEL,
    image_file_ids: dict[str, str] | None = None,
) -> int:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    row_count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for sample in samples:
            image_file_id = (image_file_ids or {}).get(sample.image_id)
            if not image_file_id:
                raise ValueError(f"Missing image_file_id for sample {sample.image_id}")
            request_item = build_openai_tag_request(
                sample,
                model=model,
                image_file_id=image_file_id,
            )
            handle.write(json.dumps(request_item.to_batch_json()))
            handle.write("\n")
            row_count += 1
    return row_count


def ensure_uploaded_image_files(
    samples: Iterable[SampleRow],
    *,
    cache_path: str,
    api_key_env: str = "OPENAI_API_KEY",
    transport: OpenAIBatchTransport | None = None,
) -> dict[str, str]:
    api_key = os.environ.get(api_key_env)
    if transport is None:
        if not api_key:
            raise ValueError(f"{api_key_env} is required for ensure_uploaded_image_files")
        transport = OpenAIBatchTransport(api_key)

    cache_file = Path(cache_path)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached = _read_uploaded_image_cache(cache_file)
    header_needed = not cache_file.exists()
    if header_needed:
        with cache_file.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["image_id", "file_path", "file_id"])
            writer.writeheader()

    with cache_file.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "file_path", "file_id"])
        for sample in samples:
            existing = cached.get(sample.image_id)
            if existing and existing.file_path == sample.file_path and existing.file_id:
                continue
            upload = transport.upload_file(sample.file_path, purpose="vision")
            file_id = str(upload.get("id", "")).strip()
            if not file_id:
                raise RuntimeError(f"OpenAI vision upload did not return a file id for {sample.image_id}")
            cached[sample.image_id] = UploadedImageFile(
                image_id=sample.image_id,
                file_path=sample.file_path,
                file_id=file_id,
            )
            writer.writerow(
                {
                    "image_id": sample.image_id,
                    "file_path": sample.file_path,
                    "file_id": file_id,
                }
            )
            handle.flush()
    return {image_id: item.file_id for image_id, item in cached.items()}


def submit_batch(
    *,
    batch_input_path: str,
    api_key_env: str = "OPENAI_API_KEY",
    transport: OpenAIBatchTransport | None = None,
) -> OpenAIBatchJob:
    api_key = os.environ.get(api_key_env)
    if transport is None:
        if not api_key:
            raise ValueError(f"{api_key_env} is required for submit_batch")
        transport = OpenAIBatchTransport(api_key)
    upload = transport.upload_file(batch_input_path)
    input_file_id = str(upload.get("id", "")).strip()
    if not input_file_id:
        raise RuntimeError("OpenAI file upload did not return a file id")
    payload = transport.create_batch(input_file_id)
    return OpenAIBatchJob(
        job_id=str(payload.get("id", "")).strip(),
        input_file_id=input_file_id,
        output_file_id=_optional_str(payload.get("output_file_id")),
        error_file_id=_optional_str(payload.get("error_file_id")),
        status=str(payload.get("status", "")).strip(),
    )


def retrieve_batch_job(
    batch_id: str,
    *,
    api_key_env: str = "OPENAI_API_KEY",
    transport: OpenAIBatchTransport | None = None,
) -> OpenAIBatchJob:
    api_key = os.environ.get(api_key_env)
    if transport is None:
        if not api_key:
            raise ValueError(f"{api_key_env} is required for retrieve_batch_job")
        transport = OpenAIBatchTransport(api_key)
    payload = transport.retrieve_batch(batch_id)
    return OpenAIBatchJob(
        job_id=str(payload.get("id", "")).strip(),
        input_file_id=str(payload.get("input_file_id", "")).strip(),
        output_file_id=_optional_str(payload.get("output_file_id")),
        error_file_id=_optional_str(payload.get("error_file_id")),
        status=str(payload.get("status", "")).strip(),
    )


def collect_batch_results(
    *,
    job: OpenAIBatchJob,
    samples_by_id: dict[str, SampleRow],
    output_path: str,
    errors_path: str,
    raw_output_log_dir: str | None = None,
    api_key_env: str = "OPENAI_API_KEY",
    transport: OpenAIBatchTransport | None = None,
) -> tuple[list[RawTagRow], int]:
    if not job.output_file_id:
        raise ValueError("Batch job does not have an output file id")
    api_key = os.environ.get(api_key_env)
    if transport is None:
        if not api_key:
            raise ValueError(f"{api_key_env} is required for collect_batch_results")
        transport = OpenAIBatchTransport(api_key)

    output_text = transport.download_file_text(job.output_file_id)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(output_text, encoding="utf-8")

    error_text = ""
    if job.error_file_id:
        error_text = transport.download_file_text(job.error_file_id)
    Path(errors_path).parent.mkdir(parents=True, exist_ok=True)
    Path(errors_path).write_text(error_text, encoding="utf-8")

    rows: list[RawTagRow] = []
    invalid_count = 0
    for line in output_text.splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        custom_id = str(payload.get("custom_id", "")).strip()
        sample = samples_by_id.get(custom_id)
        if sample is None:
            continue
        response_body = payload.get("response", {}).get("body", {})
        tagging_result = parse_openai_tag_response(
            sample=sample,
            response=response_body,
            raw_output_log_dir=raw_output_log_dir,
        )
        row = build_raw_tag_row(sample, tagging_result)
        if row.review_needed.lower() == "true" and row.confidence_note == "Model output was not valid JSON":
            invalid_count += 1
        rows.append(row)
    return rows, invalid_count


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_uploaded_image_cache(path: Path) -> dict[str, UploadedImageFile]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row["image_id"]).strip(): UploadedImageFile(
                image_id=str(row["image_id"]).strip(),
                file_path=str(row["file_path"]).strip(),
                file_id=str(row["file_id"]).strip(),
            )
            for row in reader
            if str(row.get("image_id", "")).strip() and str(row.get("file_id", "")).strip()
        }


def _sleep_before_retry(base_seconds: float, attempt: int) -> None:
    time.sleep(base_seconds * (2**attempt))
