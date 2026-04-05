"""OpenAI-backed tagging helpers for archive preprocessing."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol
from urllib import error, request

from .local_vlm_tagger import DEFAULT_PROMPT, _coerce_json
from .preprocessing import RawTagRow, SampleRow, TaggingResult

DEFAULT_OPENAI_VISION_MODEL = "gpt-4.1-mini"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_FILES_URL = "https://api.openai.com/v1/files"
OPENAI_BATCHES_URL = "https://api.openai.com/v1/batches"


class HTTPTransport(Protocol):
    def post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        """Send JSON to the API and return parsed JSON."""


@dataclass(slots=True)
class OpenAITagRequest:
    custom_id: str
    sample: SampleRow
    payload: dict[str, object]

    def to_batch_json(self) -> dict[str, object]:
        return {
            "custom_id": self.custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": self.payload,
        }


class OpenAIHTTPTransport:
    """Tiny HTTP client so preprocessing can use OpenAI without extra deps."""

    def __init__(
        self,
        api_key: str,
        timeout_seconds: float = 180.0,
        max_retries: int = 5,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        retryable_http_codes = {429, 500, 502, 503, 504, 520}
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            http_request = request.Request(url=url, method="POST", data=body, headers=headers)
            try:
                with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
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


@dataclass(slots=True)
class OpenAIJsonTagger:
    model: str = DEFAULT_OPENAI_VISION_MODEL
    prompt: str = DEFAULT_PROMPT
    api_key_env: str = "OPENAI_API_KEY"
    transport: HTTPTransport | None = None
    raw_output_log_dir: str | None = None

    def __post_init__(self) -> None:
        if self.transport is None:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise ValueError(f"{self.api_key_env} is required for OpenAIJsonTagger")
            self.transport = OpenAIHTTPTransport(api_key)

    def tag_image(self, sample: SampleRow) -> TaggingResult:
        request_item = build_openai_tag_request(
            sample,
            model=self.model,
            prompt=self.prompt,
        )
        assert self.transport is not None
        response = self.transport.post_json(OPENAI_RESPONSES_URL, request_item.payload)
        return parse_openai_tag_response(
            sample=sample,
            response=response,
            raw_output_log_dir=self.raw_output_log_dir,
        )


def build_openai_tag_request(
    sample: SampleRow,
    *,
    model: str = DEFAULT_OPENAI_VISION_MODEL,
    prompt: str = DEFAULT_PROMPT,
    image_file_id: str | None = None,
) -> OpenAITagRequest:
    image_part = (
        _build_image_file_part(image_file_id)
        if image_file_id
        else _build_image_part(Path(sample.file_path))
    )
    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    image_part,
                ],
            }
        ],
        "text": {"format": {"type": "json_object"}},
    }
    return OpenAITagRequest(
        custom_id=sample.image_id,
        sample=sample,
        payload=payload,
    )


def build_batch_requests(
    samples: list[SampleRow],
    *,
    model: str = DEFAULT_OPENAI_VISION_MODEL,
    prompt: str = DEFAULT_PROMPT,
    image_file_ids: dict[str, str] | None = None,
) -> list[OpenAITagRequest]:
    return [
        build_openai_tag_request(
            sample,
            model=model,
            prompt=prompt,
            image_file_id=(image_file_ids or {}).get(sample.image_id),
        )
        for sample in samples
    ]


def parse_openai_tag_response(
    *,
    sample: SampleRow,
    response: dict[str, object] | str,
    raw_output_log_dir: str | None = None,
) -> TaggingResult:
    response_text = response if isinstance(response, str) else extract_openai_output_text(response)
    parsed = parse_openai_tag_response_text(
        response_text,
        image_path=sample.file_path,
        raw_output_log_dir=raw_output_log_dir,
    )
    return TaggingResult(
        caption=str(parsed.get("caption", "")).strip(),
        category=str(parsed.get("category", "")).strip(),
        silhouette=str(parsed.get("silhouette", "")).strip(),
        color=str(parsed.get("color", "")).strip(),
        material=str(parsed.get("material", "")).strip(),
        pattern=str(parsed.get("pattern", "")).strip(),
        texture=str(parsed.get("texture", "")).strip(),
        mood=str(parsed.get("mood", "")).strip(),
        season=str(parsed.get("season", "")).strip(),
        era=str(parsed.get("era", "")).strip(),
        detail=str(parsed.get("detail", "")).strip(),
        review_needed=bool(parsed.get("review_needed", False)),
        confidence_note=str(parsed.get("confidence_note", "")).strip(),
    )


def parse_openai_tag_response_text(
    response_text: str,
    *,
    image_path: str | None = None,
    raw_output_log_dir: str | None = None,
) -> dict[str, object]:
    return _coerce_json(
        response_text,
        image_path=image_path,
        raw_output_log_dir=raw_output_log_dir,
    )


def extract_openai_output_text(response: dict[str, object]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    chunks: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if not isinstance(content, dict):
                continue
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if not chunks:
        raise RuntimeError("No text content returned by OpenAI response")
    return "".join(chunks)


def build_raw_tag_row(sample: SampleRow, tagging_result: TaggingResult) -> RawTagRow:
    return RawTagRow(
        **asdict(sample),
        caption=tagging_result.caption,
        raw_category=tagging_result.category,
        raw_silhouette=tagging_result.silhouette,
        raw_color=tagging_result.color,
        raw_material=tagging_result.material,
        raw_pattern=tagging_result.pattern,
        raw_texture=tagging_result.texture,
        raw_mood=tagging_result.mood,
        raw_season=tagging_result.season,
        raw_era=tagging_result.era,
        raw_detail=tagging_result.detail,
        review_needed="true" if tagging_result.review_needed else "false",
        confidence_note=tagging_result.confidence_note,
    )


def _build_image_part(image_path: Path) -> dict[str, str]:
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        mime_type = "image/png"
    elif suffix == ".webp":
        mime_type = "image/webp"
    else:
        mime_type = "image/jpeg"
    return {
        "type": "input_image",
        "image_url": f"data:{mime_type};base64,{encoded}",
        "detail": "low",
    }


def _build_image_file_part(file_id: str) -> dict[str, str]:
    return {
        "type": "input_image",
        "file_id": file_id,
        "detail": "low",
    }


def _sleep_before_retry(base_seconds: float, attempt: int) -> None:
    time.sleep(base_seconds * (2**attempt))
