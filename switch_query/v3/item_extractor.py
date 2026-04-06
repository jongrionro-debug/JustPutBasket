"""Interfaces and helpers for V3 item extraction."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Protocol
from urllib import error, request

from switch_query.tagging.preprocessing import NormalizedTagRow

from .models import (
    V3ArchiveDocument,
    V3DocumentItem,
    V3ItemExtractionInput,
    V3ItemExtractionOutput,
)

LUXIA_OPENAI_CHAT_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o/create"
DEFAULT_IMAGE_TRANSFER_MODE = "safe_resize"


class ItemExtractor(Protocol):
    """Protocol for any V3 item extractor backend."""

    def extract_items(self, extraction_input: V3ItemExtractionInput) -> V3ItemExtractionOutput:
        """Return item-level enrichment for one archive document."""


class LuxiaTransport(Protocol):
    """Minimal HTTP transport for Luxia OpenAI-style calls."""

    def post_json(self, url: str, payload: dict[str, object], *, api_key: str) -> dict[str, object]:
        """Send JSON to Luxia and return parsed JSON."""


@dataclass(slots=True)
class LuxiaHTTPTransport:
    """Tiny stdlib HTTP client for Luxia OpenAI-style requests."""

    timeout_seconds: float = 180.0

    def post_json(self, url: str, payload: dict[str, object], *, api_key: str) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            url=url,
            method="POST",
            data=body,
            headers={
                "apikey": api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Luxia request failed: {exc.code} {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Luxia request failed: {exc}") from exc


@dataclass(slots=True)
class LuxiaItemExtractorConfig:
    api_key_env: str = "LUXIA_API_KEY"
    api_url: str = LUXIA_OPENAI_CHAT_URL
    model: str = "gpt-4o-2024-08-06"
    image_transfer_mode: str = DEFAULT_IMAGE_TRANSFER_MODE
    max_image_edge: int = 1024
    jpeg_quality: int = 85


@dataclass(slots=True)
class LuxiaItemExtractor:
    """Luxia-backed item extractor using OpenAI-style chat completions."""

    config: LuxiaItemExtractorConfig = field(default_factory=LuxiaItemExtractorConfig)
    transport: LuxiaTransport | None = None

    def __post_init__(self) -> None:
        if self.transport is None:
            self.transport = LuxiaHTTPTransport()

    def extract_items(self, extraction_input: V3ItemExtractionInput) -> V3ItemExtractionOutput:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.config.api_key_env} is required for Luxia item extraction.")

        payload = build_luxia_item_extraction_request(
            extraction_input,
            model=self.config.model,
            image_transfer_mode=self.config.image_transfer_mode,
            max_image_edge=self.config.max_image_edge,
            jpeg_quality=self.config.jpeg_quality,
        )
        assert self.transport is not None
        response = self.transport.post_json(self.config.api_url, payload, api_key=api_key)
        return parse_luxia_item_extraction_response(
            response,
            image_id=extraction_input.image_id,
            extraction_mode=extraction_input.extraction_mode,
        )


def extraction_input_from_normalized_row(
    row: NormalizedTagRow,
    *,
    extraction_mode: str = "text_only",
) -> V3ItemExtractionInput:
    """Build a normalized extractor input from a canonicalized V2 row."""

    return V3ItemExtractionInput(
        image_id=row.image_id,
        file_path=row.file_path,
        brand=row.brand,
        season_group=row.season_group,
        canonical_tags={
            "category": row.canonical_category,
            "silhouette": row.canonical_silhouette,
            "color": row.canonical_color,
            "material": row.canonical_material,
            "pattern": row.canonical_pattern,
            "texture": row.canonical_texture,
            "mood": row.canonical_mood,
            "season": row.canonical_season,
            "era": row.canonical_era,
            "detail": row.canonical_detail,
        },
        raw_tags={
            "category": row.raw_category,
            "silhouette": row.raw_silhouette,
            "color": row.raw_color,
            "material": row.raw_material,
            "pattern": row.raw_pattern,
            "texture": row.raw_texture,
            "mood": row.raw_mood,
            "season": row.raw_season,
            "era": row.raw_era,
            "detail": row.raw_detail,
        },
        detail=row.canonical_detail or row.raw_detail,
        image_path=row.file_path,
        extraction_mode=extraction_mode,  # type: ignore[arg-type]
    )


def apply_item_extraction(
    document: V3ArchiveDocument,
    extraction_output: V3ItemExtractionOutput,
) -> V3ArchiveDocument:
    """Return a document enriched with extracted item metadata."""

    return replace(
        document,
        items=list(extraction_output.items),
        item_confidence=extraction_output.item_confidence,
        item_extraction_notes=list(extraction_output.item_extraction_notes),
    )


def build_luxia_item_extraction_request(
    extraction_input: V3ItemExtractionInput,
    *,
    model: str,
    image_transfer_mode: str = DEFAULT_IMAGE_TRANSFER_MODE,
    max_image_edge: int = 1024,
    jpeg_quality: int = 85,
) -> dict[str, object]:
    user_content: list[dict[str, object]] = [{"type": "text", "text": _build_extraction_prompt(extraction_input)}]
    if extraction_input.extraction_mode == "image_assisted" and extraction_input.image_path:
        user_content.append(
            _build_image_url_part(
                extraction_input.image_path,
                image_transfer_mode=image_transfer_mode,
                max_image_edge=max_image_edge,
                jpeg_quality=jpeg_quality,
            )
        )

    return {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are extracting item-level fashion structure from one runway look. "
                    "Return compact JSON only and no prose. "
                    "Do not include empty arrays, nulls, evidence, source, or notes."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ],
        "stream": False,
    }


def parse_luxia_item_extraction_response(
    response: dict[str, object],
    *,
    image_id: str = "",
    extraction_mode: str = "text_only",
) -> V3ItemExtractionOutput:
    response_text = _extract_luxia_text(response)
    payload = _load_json_payload(response_text)

    if "items" not in payload:
        raise RuntimeError("Luxia item extraction output must contain an items list.")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise RuntimeError("Luxia item extraction output must contain an items list.")

    default_source = (
        "luxia_image_assisted" if extraction_mode == "image_assisted" else "luxia_text_only"
    )
    items: list[V3DocumentItem] = []
    for index, raw_item in enumerate(raw_items, start=1):
        if not isinstance(raw_item, dict):
            continue
        items.append(
            V3DocumentItem(
                item_id=_normalized_item_id(image_id=image_id, index=index),
                category=_coerce_string(raw_item.get("category")),
                color=_coerce_string_list(raw_item.get("color")),
                silhouette=_coerce_string_list(raw_item.get("silhouette")),
                material=_coerce_string_list(raw_item.get("material")),
                pattern=_coerce_string_list(raw_item.get("pattern")),
                texture=_coerce_string_list(raw_item.get("texture")),
                style_tags=_coerce_string_list(raw_item.get("style_tags")),
                confidence=_coerce_float(raw_item.get("confidence")),
                evidence=_coerce_string_list(raw_item.get("evidence")),
                source=_coerce_string(raw_item.get("source")) or default_source,
            )
        )

    return V3ItemExtractionOutput(
        items=items,
        item_confidence=_coerce_float(payload.get("item_confidence")),
        item_extraction_notes=_coerce_string_list(payload.get("item_extraction_notes")),
    )


def _build_extraction_prompt(extraction_input: V3ItemExtractionInput) -> str:
    canonical_tags = _compact_tag_dict(extraction_input.canonical_tags)
    raw_tags = _compact_tag_dict(extraction_input.raw_tags)
    context_lines = [
        "Extract item-level fashion structure from one runway look.",
        'Return only valid compact JSON with this shape: {"items":[{"category":"","confidence":0.0}],"item_confidence":0.0}',
        "Optional item keys only when non-empty: color, silhouette, material, pattern, texture, style_tags.",
        "Split the look into individual wearable items when possible.",
        "Attach color and silhouette to the correct item, not to the whole outfit.",
        "If an attribute belongs to a different item, do not copy it across items.",
        "If uncertain, leave the attribute empty or lower the confidence.",
        "Do not include evidence, source, item_extraction_notes, empty arrays, or null values.",
        "",
        f"image_id: {extraction_input.image_id}",
        f"brand: {extraction_input.brand}",
        f"season_group: {extraction_input.season_group}",
        f"detail: {extraction_input.detail}",
        f"canonical_tags: {json.dumps(canonical_tags, ensure_ascii=False)}",
        f"raw_tags: {json.dumps(raw_tags, ensure_ascii=False)}",
        f"extraction_mode: {extraction_input.extraction_mode}",
        "",
        "Each item must include category and confidence.",
        "Keep values short and use arrays only when an attribute has multiple distinct values.",
    ]
    return "\n".join(context_lines)


def _build_image_url_part(
    image_path: str,
    *,
    image_transfer_mode: str,
    max_image_edge: int,
    jpeg_quality: int,
) -> dict[str, object]:
    mime_type, image_bytes = _prepare_image_bytes_for_transfer(
        image_path,
        image_transfer_mode=image_transfer_mode,
        max_image_edge=max_image_edge,
        jpeg_quality=jpeg_quality,
    )
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
        },
    }


def _prepare_image_bytes_for_transfer(
    image_path: str,
    *,
    image_transfer_mode: str,
    max_image_edge: int,
    jpeg_quality: int,
) -> tuple[str, bytes]:
    image_file = Path(image_path)
    raw_bytes = image_file.read_bytes()
    if image_transfer_mode != DEFAULT_IMAGE_TRANSFER_MODE:
        return _mime_type_for_image(image_file), raw_bytes

    if max_image_edge <= 0:
        return _mime_type_for_image(image_file), raw_bytes

    try:
        from PIL import Image
    except ImportError:
        return _mime_type_for_image(image_file), raw_bytes

    try:
        with Image.open(image_file) as image:
            image.load()
            if max(image.size) <= max_image_edge:
                return _mime_type_for_image(image_file), raw_bytes

            resized = image.copy()
            resized.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
            if resized.mode not in {"RGB", "L"}:
                background = Image.new("RGB", resized.size, (255, 255, 255))
                alpha_source = resized.convert("RGBA")
                background.paste(alpha_source, mask=alpha_source.getchannel("A"))
                resized = background
            elif resized.mode == "L":
                resized = resized.convert("RGB")

            buffer = io.BytesIO()
            resized.save(
                buffer,
                format="JPEG",
                quality=max(min(jpeg_quality, 95), 40),
                optimize=True,
            )
            return "image/jpeg", buffer.getvalue()
    except Exception:
        return _mime_type_for_image(image_file), raw_bytes


def _mime_type_for_image(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"
    return "image/jpeg"


def _extract_luxia_text(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message", {})
            if isinstance(message, dict):
                return _coerce_message_content(message.get("content"))
    if "content" in response:
        return _coerce_message_content(response.get("content"))
    raise RuntimeError("Luxia response did not contain parseable text content.")


def _coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        collapsed = "\n".join(part.strip() for part in parts if part.strip())
        if collapsed:
            return collapsed
    raise RuntimeError("Luxia message content was empty or unsupported.")


def _load_json_payload(response_text: str) -> dict[str, object]:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fences(cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = _extract_first_json_object(cleaned)
        if payload is None:
            raise RuntimeError("Luxia item extraction output was not valid JSON.")

    if not isinstance(payload, dict):
        raise RuntimeError("Luxia item extraction output must be a JSON object.")
    return payload


def _extract_first_json_object(text: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    candidate_starts = [index for index, char in enumerate(text) if char == "{"]
    for start in candidate_starts:
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :]
    elif stripped.startswith("```"):
        stripped = stripped[len("```") :]
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")]
    return stripped.strip()


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _coerce_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def _compact_tag_dict(values: dict[str, str]) -> dict[str, str]:
    return {
        key: value.strip()
        for key, value in values.items()
        if isinstance(value, str) and value.strip()
    }


def _normalized_item_id(*, image_id: str, index: int) -> str:
    prefix = image_id.strip()
    if prefix:
        return f"{prefix}#{index}"
    return f"item_{index}"


# Compatibility aliases to avoid breaking existing integrations during migration.
GeminiHTTPTransport = LuxiaHTTPTransport
GeminiItemExtractorConfig = LuxiaItemExtractorConfig
GeminiItemExtractor = LuxiaItemExtractor


def build_gemini_item_extraction_request(
    extraction_input: V3ItemExtractionInput,
    *,
    model: str,
) -> dict[str, object]:
    return build_luxia_item_extraction_request(extraction_input, model=model)


def parse_gemini_item_extraction_response(
    response: dict[str, object],
    *,
    image_id: str = "",
    extraction_mode: str = "image_assisted",
) -> V3ItemExtractionOutput:
    return parse_luxia_item_extraction_response(
        response,
        image_id=image_id,
        extraction_mode=extraction_mode,
    )
