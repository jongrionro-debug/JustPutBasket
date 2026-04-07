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

from .concepts import (
    COLOR_TOKENS,
    extract_color_style_tags,
    extract_style_concepts,
    is_apparel_category,
    normalize_token,
    split_multi_value_text,
)
from .models import (
    V3ArchiveDocument,
    V3DocumentItem,
    V3ItemExtractionInput,
    V3ItemExtractionOutput,
)

LUXIA_OPENAI_CHAT_URL = "https://bridge.luxiacloud.com/llm/openai/chat/completions/gpt-4o/create"
DEFAULT_IMAGE_TRANSFER_MODE = "safe_resize"
CONTEXT_ENRICHMENT_NOTE = "items enriched from canonical/raw mood-era-detail context"
COLOR_REQUIRED_CATEGORY_HINTS = frozenset(
    {
        "bag",
        "blazer",
        "blouse",
        "boots",
        "cardigan",
        "coat",
        "dress",
        "handbag",
        "hat",
        "jacket",
        "jeans",
        "pants",
        "sandals",
        "shirt",
        "shoes",
        "shorts",
        "skirt",
        "sweater",
        "top",
        "trousers",
        "vest",
    }
)


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
        extraction_output = parse_luxia_item_extraction_response(
            response,
            image_id=extraction_input.image_id,
            extraction_mode=extraction_input.extraction_mode,
        )
        return _enrich_items_from_input(extraction_output, extraction_input)


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
                    "Keep item-level attributes tied to the correct visible item. "
                    "Do not include empty arrays, nulls, source, or notes."
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
                style_concepts=_coerce_style_concept_list(raw_item.get("style_concepts")),
                confidence=_coerce_float(raw_item.get("confidence")),
                evidence=_coerce_string_list(raw_item.get("evidence")),
                source=_coerce_string(raw_item.get("source")) or default_source,
            )
        )

    return V3ItemExtractionOutput(
        items=_normalize_item_style_fields(items),
        item_confidence=_coerce_float(payload.get("item_confidence")),
        item_extraction_notes=_coerce_string_list(payload.get("item_extraction_notes")),
    )


def _build_extraction_prompt(extraction_input: V3ItemExtractionInput) -> str:
    canonical_tags = _compact_tag_dict(extraction_input.canonical_tags)
    raw_tags = _compact_tag_dict(extraction_input.raw_tags)
    context_lines = [
        "Extract item-level fashion structure from one runway look.",
        'Return only valid compact JSON with this shape: {"items":[{"category":"","confidence":0.0}],"item_confidence":0.0}',
        "Optional item keys only when non-empty: color, silhouette, material, pattern, texture, style_tags, style_concepts, evidence.",
        "",
        "Follow this exact process internally before you answer:",
        "1. Identify every visible wearable item in the look.",
        "2. Assign one category to each visible item.",
        "3. Fill item-specific attributes for each item only.",
        "4. Separate style_tags from style_concepts.",
        "5. Return compact JSON only.",
        "",
        "Priority of evidence:",
        "1. Trust the image first.",
        "2. Use detail text as secondary support.",
        "3. Use canonical_tags and raw_tags only as weak hints.",
        "",
        "Definitions:",
        "style_concepts = high-level search concepts such as vintage, minimal, romantic, avant-garde, retro.",
        "style_tags = item-specific descriptors such as peep-toe, double-breasted, cropped, lace-up, open-toe, embellished.",
        "color = visible dominant color(s) of this item only, maximum 2 values.",
        "",
        "Hard rules:",
        "Never copy outfit-level color to every item.",
        "Never copy one item's attributes onto another item.",
        "Never put vintage, minimal, romantic, avant-garde, retro, edgy, elegant, modern, sporty into style_tags.",
        "If bag, shoes, belt, hat, or other visible accessories are present, keep them as separate items.",
        "For clearly visible apparel, bags, and shoes, always predict at least one dominant color.",
        "Leave color empty only when the item is heavily occluded, highly transparent, metallic-reflective, or genuinely indeterminate.",
        "If pattern or texture is visible but color is not reliable, keep pattern or texture and leave color empty.",
        "Use arrays only when an attribute truly has multiple distinct values.",
        "Keep values short and literal.",
        "",
        "Field guidance:",
        "category: required for every item.",
        "confidence: required for every item.",
        "evidence: short visual phrase for why this item exists or why a key attribute was chosen.",
        "For top, dress, jacket, coat, shirt, blouse, knit, trousers, pants, skirt, shorts, bag, handbag, shoes, boots, and sandals, color should usually be present.",
        "pattern examples: snakeskin, floral, plaid, striped, studs.",
        "texture examples: feathered, sheer, ribbed, smooth, quilted.",
        "material examples: leather, satin, denim, wool, lace, knit.",
        "",
        "Do not include source or item_extraction_notes in the JSON.",
        "Do not include empty arrays or null values unless the field is required by the schema example.",
        "",
        "Few-shot guidance:",
        'Example A input detail: "pink camisole|snakeskin midi skirt|red leather handbag"',
        'Example A output idea: top.color=["pink"], skirt.pattern=["snakeskin"], bag.color=["red"]',
        'Example B input detail: "vintage washed jacket|black trousers"',
        'Example B output idea: jacket.style_concepts=["vintage"], trousers.color=["black"]',
        'Example C input detail: "white top|black skirt|two-tone heels"',
        'Example C output idea: shoes.style_tags=["two-tone"] and shoes.color may contain white and black if both are clearly visible',
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
        "Return JSON only. No markdown. No prose.",
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


def _coerce_style_concept_list(value: object) -> list[str]:
    concepts = _coerce_string_list(value)
    inferred: list[str] = []
    for concept in concepts:
        extracted = extract_style_concepts(concept)
        if extracted:
            for extracted_concept in extracted:
                if extracted_concept not in inferred:
                    inferred.append(extracted_concept)
            continue
        normalized = normalize_token(concept)
        if normalized and normalized not in inferred:
            inferred.append(normalized)
    return inferred


def _normalize_item_style_fields(items: list[V3DocumentItem]) -> list[V3DocumentItem]:
    normalized_items: list[V3DocumentItem] = []
    for item in items:
        migrated_concepts = list(item.style_concepts)
        normalized_style_tags: list[str] = []
        for style_tag in item.style_tags:
            extracted_concepts = extract_style_concepts(style_tag)
            if extracted_concepts:
                for style_concept in extracted_concepts:
                    if style_concept not in migrated_concepts:
                        migrated_concepts.append(style_concept)
                continue
            normalized_tag = normalize_token(style_tag)
            if normalized_tag and normalized_tag not in normalized_style_tags:
                normalized_style_tags.append(normalized_tag)
        normalized_items.append(
            replace(
                item,
                style_tags=normalized_style_tags,
                style_concepts=migrated_concepts,
            )
        )
    return normalized_items


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


def _enrich_items_from_input(
    extraction_output: V3ItemExtractionOutput,
    extraction_input: V3ItemExtractionInput,
) -> V3ItemExtractionOutput:
    inferred_style_concepts = _infer_style_concepts_from_input(extraction_input)
    inferred_style_tags = _infer_style_tags_from_input(extraction_input)
    canonical_colors = split_multi_value_text(extraction_input.canonical_tags.get("color", ""))
    updated_items = []
    enriched_count = 0
    apparel_item_count = sum(
        1 for item in extraction_output.items if is_apparel_category(item.category)
    )
    missing_color_count = sum(1 for item in extraction_output.items if not item.color)
    for item in extraction_output.items:
        merged_style_tags = list(item.style_tags)
        for style_tag in inferred_style_tags:
            if style_tag not in merged_style_tags:
                merged_style_tags.append(style_tag)

        merged_style_concepts = list(item.style_concepts)
        for style_concept in inferred_style_concepts:
            if style_concept not in merged_style_concepts:
                merged_style_concepts.append(style_concept)

        merged_colors = list(item.color)
        if not merged_colors:
            merged_colors = _infer_item_colors_from_input(
                item,
                extraction_input=extraction_input,
                canonical_colors=canonical_colors,
                apparel_item_count=apparel_item_count,
                missing_color_count=missing_color_count,
            )

        if (
            merged_style_tags != item.style_tags
            or merged_style_concepts != item.style_concepts
            or merged_colors != item.color
        ):
            enriched_count += 1
        updated_items.append(
            replace(
                item,
                color=merged_colors,
                style_tags=merged_style_tags,
                style_concepts=merged_style_concepts,
            )
        )

    if enriched_count == 0:
        return extraction_output
    notes = list(extraction_output.item_extraction_notes)
    if CONTEXT_ENRICHMENT_NOTE not in notes:
        notes.append(CONTEXT_ENRICHMENT_NOTE)
    return V3ItemExtractionOutput(
        items=updated_items,
        item_confidence=extraction_output.item_confidence,
        item_extraction_notes=notes,
    )


def _infer_style_concepts_from_input(extraction_input: V3ItemExtractionInput) -> list[str]:
    context_parts = [
        extraction_input.detail,
        extraction_input.canonical_tags.get("mood", ""),
        extraction_input.canonical_tags.get("era", ""),
        extraction_input.raw_tags.get("mood", ""),
        extraction_input.raw_tags.get("era", ""),
        extraction_input.canonical_tags.get("detail", ""),
        extraction_input.raw_tags.get("detail", ""),
    ]
    context = " ".join(part.strip() for part in context_parts if part and part.strip())
    return extract_style_concepts(context)


def _infer_style_tags_from_input(extraction_input: V3ItemExtractionInput) -> list[str]:
    context_parts = [
        extraction_input.detail,
        extraction_input.canonical_tags.get("detail", ""),
        extraction_input.raw_tags.get("detail", ""),
    ]
    context = " ".join(part.strip() for part in context_parts if part and part.strip())
    return extract_color_style_tags(context)


def _infer_item_colors_from_input(
    item: V3DocumentItem,
    *,
    extraction_input: V3ItemExtractionInput,
    canonical_colors: list[str],
    apparel_item_count: int,
    missing_color_count: int,
) -> list[str]:
    detail_colors = _infer_colors_from_detail_phrase(item.category, extraction_input.detail)
    if detail_colors:
        return detail_colors[:2]
    evidence_colors = _infer_colors_from_evidence(item.evidence)
    if evidence_colors:
        return evidence_colors[:2]
    if (
        _color_should_usually_be_present(item.category)
        and len(canonical_colors) == 1
        and (
            apparel_item_count == 1
            or missing_color_count == 1
            or _is_accessory_like(item.category)
        )
    ):
        return canonical_colors[:1]
    if _color_should_usually_be_present(item.category) and canonical_colors:
        return canonical_colors[:1]
    return []


def _infer_colors_from_detail_phrase(category: str, detail: str) -> list[str]:
    normalized_category = normalize_token(category)
    category_tokens = {
        normalized_category,
        normalized_category.rstrip("s"),
        f"{normalized_category}s",
    }
    for phrase in split_multi_value_text(detail):
        normalized_phrase = normalize_token(phrase)
        if not any(token and token in normalized_phrase for token in category_tokens):
            continue
        colors = [
            color for color in COLOR_TOKENS if f" {color} " in f" {normalized_phrase} "
        ]
        if colors:
            deduped: list[str] = []
            for color in colors:
                if color not in deduped:
                    deduped.append(color)
            return deduped[:2]
    return []


def _infer_colors_from_evidence(evidence: list[str]) -> list[str]:
    deduped: list[str] = []
    for phrase in evidence:
        normalized_phrase = normalize_token(phrase)
        for color in COLOR_TOKENS:
            if f" {color} " in f" {normalized_phrase} " and color not in deduped:
                deduped.append(color)
    return deduped[:2]


def _color_should_usually_be_present(category: str) -> bool:
    return normalize_token(category) in COLOR_REQUIRED_CATEGORY_HINTS


def _is_accessory_like(category: str) -> bool:
    return normalize_token(category) in {"bag", "handbag", "hat", "shoes", "boots", "sandals"}


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
