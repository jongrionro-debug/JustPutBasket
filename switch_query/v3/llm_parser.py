"""Luxia-backed structured query parser for the V3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
import sys
from typing import Any

import requests

from .models import V3ParsedQuery, V3TargetItem
from .query_validation import ensure_valid_v3_parsed_query

LUXIA_CHAT_URL = "https://bridge.luxiacloud.com/luxia/v1/chat"
ITEM_ATTRIBUTE_NAMES = (
    "category",
    "color",
    "silhouette",
    "material",
    "pattern",
    "texture",
    "style_tags",
)
ATTRIBUTE_PRIORITY_NAMES = tuple(name for name in ITEM_ATTRIBUTE_NAMES if name != "category")
ITEM_STYLE_TAG_ALIASES = ("style_tags", "styles", "style")
COLOR_PHRASES = (
    "charcoal",
    "burgundy",
    "lavender",
    "maroon",
    "silver",
    "purple",
    "yellow",
    "orange",
    "white",
    "black",
    "beige",
    "brown",
    "camel",
    "cream",
    "green",
    "khaki",
    "ivory",
    "olive",
    "navy",
    "blue",
    "pink",
    "gray",
    "grey",
    "gold",
    "red",
    "tan",
)
SILHOUETTE_PHRASES = (
    "wide leg",
    "straight leg",
    "slim fit",
    "wide-leg",
    "straight-leg",
    "slim-fit",
    "oversized",
    "structured",
    "relaxed",
    "tailored",
    "fitted",
    "cropped",
    "fluid",
    "loose",
)
MATERIAL_PHRASES = (
    "leather",
    "denim",
    "wool",
    "silk",
    "lace",
    "satin",
    "linen",
    "cotton",
    "jersey",
    "suede",
    "knit",
)
PATTERN_PHRASES = (
    "polka dot",
    "lace trim",
    "striped",
    "checked",
    "floral",
    "plaid",
)
TEXTURE_PHRASES = (
    "feathered",
    "feathers",
    "pleated",
    "quilted",
    "smooth",
    "sheer",
    "ribbed",
)
STYLE_TAG_PHRASES = (
    "avant garde",
    "avant-garde",
    "contemporary",
    "minimal",
    "vintage",
    "romantic",
    "elegant",
    "modern",
    "sporty",
)


@dataclass(slots=True)
class LuxiaV3QueryParserConfig:
    api_url: str = LUXIA_CHAT_URL
    api_key_env: str = "LUXIA_API_KEY"
    model: str = "luxia3-llm-32b-0731"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    max_completion_tokens: int = 1024
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    max_retries: int = 2


@dataclass(slots=True)
class LuxiaV3QueryParser:
    config: LuxiaV3QueryParserConfig = field(default_factory=LuxiaV3QueryParserConfig)
    session: requests.Session | None = None

    def parse(
        self,
        query_text: str,
        *,
        stage: str,
        balance_score: float,
        user_uploaded_image: str | None = None,
    ) -> V3ParsedQuery:
        del user_uploaded_image

        last_error: Exception | None = None
        for _attempt in range(self.config.max_retries + 1):
            try:
                api_key = os.environ.get(self.config.api_key_env)
                if not api_key:
                    raise RuntimeError(
                        f"{self.config.api_key_env} is required for Luxia query parsing."
                    )
                payload = self._post_chat(
                    query_text=query_text,
                    stage=stage,
                    balance_score=balance_score,
                    api_key=api_key,
                )
                response_text = _extract_response_text(payload)
                _debug_log("raw_response_text", response_text)
                structured_payload = _load_json_payload(response_text)
                _debug_log("structured_payload", structured_payload)
                parsed_query = _build_parsed_query(query_text, structured_payload)
                return ensure_valid_v3_parsed_query(
                    parsed_query,
                    payload=structured_payload,
                ).validated_query
            except Exception as exc:
                last_error = exc
                _debug_log("parse_error", str(exc))

        if last_error is not None:
            raise RuntimeError(f"Luxia V3 query parsing failed: {last_error}") from last_error
        raise RuntimeError("Luxia V3 query parsing failed without a captured exception")

    def _post_chat(
        self,
        *,
        query_text: str,
        stage: str,
        balance_score: float,
        api_key: str,
    ) -> Any:
        client = self.session or requests.Session()
        try:
            response = client.post(
                self.config.api_url,
                headers={
                    "apikey": api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.config.model,
                    "messages": _build_messages(
                        query_text,
                        stage=stage,
                        balance_score=balance_score,
                    ),
                    "stream": False,
                    "temperature": self.config.temperature,
                    "max_completion_tokens": self.config.max_completion_tokens,
                    "top_p": self.config.top_p,
                    "frequency_penalty": self.config.frequency_penalty,
                },
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Luxia request failed: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"Luxia request failed with status {response.status_code}: {response.text}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Luxia response was not valid JSON.") from exc


def _build_messages(query_text: str, *, stage: str, balance_score: float) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a structured fashion-query parser. "
                "Return only JSON with these keys: target_items, global_constraints, "
                "style_preferences, confidence. "
                "Each target item must contain target_item_id, category, and raw_phrase. "
                "Allowed item attribute keys are: category, color, silhouette, material, "
                "pattern, texture, style_tags, required_attributes, preferred_attributes. "
                "Do not put item attributes at the top level. "
                "For multi-item queries, split attributes across the correct items. "
                "If a raw phrase explicitly says a color, silhouette, material, pattern, texture, or style "
                "for one item, copy that attribute into that item's structured fields. "
                "Do not leave color empty when the raw phrase says things like 'white trousers' or 'black jacket'. "
                "For multi-item queries, prefer item-level binding over style_preferences for category, color, "
                "silhouette, material, pattern, and texture. "
                "Do not add markdown, prose, or explanation outside the JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"stage={stage}\n"
                f"balance_score={balance_score}\n"
                f"query={query_text}"
            ),
        },
    ]


def _extract_response_text(payload: Any) -> str:
    if isinstance(payload, dict):
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            return _coerce_message_content(message.get("content"))
        if "content" in payload:
            return _coerce_message_content(payload.get("content"))
        if "message" in payload and isinstance(payload["message"], dict):
            return _coerce_message_content(payload["message"].get("content"))
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
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        collapsed = "\n".join(part.strip() for part in parts if part.strip())
        if collapsed:
            return collapsed
    raise RuntimeError("Luxia message content was empty or unsupported.")


def _load_json_payload(response_text: str) -> dict[str, Any]:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fences(cleaned)

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise RuntimeError("Luxia parser output was not valid JSON.")
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise RuntimeError("Luxia parser output was not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise RuntimeError("Luxia parser output must be a JSON object.")
    return payload


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :]
    elif stripped.startswith("```"):
        stripped = stripped[len("```") :]
    if stripped.endswith("```"):
        stripped = stripped[: -len("```")]
    return stripped.strip()


def _build_parsed_query(query_text: str, payload: dict[str, Any]) -> V3ParsedQuery:
    target_items = _coerce_target_items(payload.get("target_items"), query_text=query_text)
    global_constraints = _coerce_named_list_map(payload.get("global_constraints"))
    style_preferences = _coerce_named_list_map(payload.get("style_preferences"))
    confidence = _coerce_confidence(payload.get("confidence", 0.0))
    parsed_query = V3ParsedQuery(
        query_text=query_text,
        target_items=target_items,
        global_constraints=global_constraints,
        style_preferences=style_preferences,
        confidence=confidence,
    )
    return _repair_query_attribute_binding(parsed_query)


def _coerce_target_items(value: Any, *, query_text: str) -> list[V3TargetItem]:
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise RuntimeError("Luxia parser output must contain a target_items list.")

    normalized: list[V3TargetItem] = []
    for index, raw_item in enumerate(value, start=1):
        if not isinstance(raw_item, dict):
            continue
        target_item_id = _coerce_string(raw_item.get("target_item_id")) or f"item_{index}"
        category = _coerce_string(raw_item.get("category"))
        raw_phrase = _coerce_string(raw_item.get("raw_phrase"))
        if not raw_phrase and len(value) == 1:
            raw_phrase = query_text

        style_values = []
        for key in ITEM_STYLE_TAG_ALIASES:
            style_values.extend(_coerce_string_list(raw_item.get(key)))

        target_item = V3TargetItem(
            target_item_id=target_item_id,
            category=category,
            color=_coerce_string_list(raw_item.get("color")),
            silhouette=_coerce_string_list(raw_item.get("silhouette")),
            material=_coerce_string_list(raw_item.get("material")),
            pattern=_coerce_string_list(raw_item.get("pattern")),
            texture=_coerce_string_list(raw_item.get("texture")),
            style_tags=_dedupe(style_values),
            required_attributes=_coerce_attribute_priority_list(raw_item.get("required_attributes")),
            preferred_attributes=_coerce_attribute_priority_list(raw_item.get("preferred_attributes")),
            raw_phrase=raw_phrase,
        )
        normalized.append(target_item)

    if not normalized:
        raise RuntimeError("Luxia parser output must contain at least one target item.")
    return normalized


def _coerce_named_list_map(value: Any) -> dict[str, list[str]]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError("Luxia parser output maps must be JSON objects.")

    normalized: dict[str, list[str]] = {}
    for raw_key, raw_value in value.items():
        key = _coerce_string(raw_key)
        if not key:
            continue
        values = _coerce_string_list(raw_value)
        if values:
            normalized[key] = values
    return normalized


def _coerce_attribute_priority_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = value.replace(",", "|").split("|")
        value = [item.strip() for item in raw_items]
    elif isinstance(value, dict):
        raw_nested = value.get("features") or value.get("values") or value.get("items")
        if raw_nested is not None:
            return _coerce_attribute_priority_list(raw_nested)
        value = [key for key, enabled in value.items() if enabled]
    if not isinstance(value, list):
        raise RuntimeError("Luxia parser attribute priority lists must be strings or arrays.")

    normalized: list[str] = []
    for raw_item in value:
        if isinstance(raw_item, dict):
            raw_item = raw_item.get("feature") or raw_item.get("name") or raw_item.get("key")
        attribute = _coerce_string(raw_item)
        if attribute in ATTRIBUTE_PRIORITY_NAMES and attribute not in normalized:
            normalized.append(attribute)
    return normalized


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = value.replace(",", "|").split("|")
    elif isinstance(value, list):
        items = [str(item) for item in value]
    else:
        raise RuntimeError("Luxia parser values must be strings or string arrays.")

    cleaned: list[str] = []
    for item in items:
        normalized = _normalize_value(item)
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


def _coerce_string(value: Any) -> str:
    if value is None:
        return ""
    return _normalize_value(str(value))


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"high", "very high"}:
            return 0.9
        if normalized in {"medium", "moderate"}:
            return 0.6
        if normalized in {"low", "very low"}:
            return 0.3
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Luxia parser confidence must be numeric.") from exc
    return max(0.0, min(1.0, confidence))


def _repair_query_attribute_binding(parsed_query: V3ParsedQuery) -> V3ParsedQuery:
    repaired_global_constraints = dict(parsed_query.global_constraints)
    repaired_style_preferences = dict(parsed_query.style_preferences)
    repaired_items: list[V3TargetItem] = []
    single_item_query = len(parsed_query.target_items) == 1

    for item in parsed_query.target_items:
        repaired_item = item

        if single_item_query:
            for feature in ITEM_ATTRIBUTE_NAMES:
                current_values = list(getattr(repaired_item, feature))
                if current_values:
                    continue

                migrated_values = repaired_style_preferences.pop(feature, None)
                if not migrated_values:
                    migrated_values = repaired_global_constraints.pop(feature, None)
                if not migrated_values:
                    continue

                repaired_item = replace(repaired_item, **{feature: migrated_values})

        repaired_item = _repair_item_attributes_from_raw_phrase(repaired_item)
        repaired_items.append(repaired_item)

    if (
        repaired_items == parsed_query.target_items
        and repaired_global_constraints == parsed_query.global_constraints
        and repaired_style_preferences == parsed_query.style_preferences
    ):
        return parsed_query

    return replace(
        parsed_query,
        target_items=repaired_items,
        global_constraints=repaired_global_constraints,
        style_preferences=repaired_style_preferences,
    )


def _repair_item_attributes_from_raw_phrase(item: V3TargetItem) -> V3TargetItem:
    phrase = _normalize_value(item.raw_phrase)
    if not phrase:
        return item

    updates: dict[str, list[str]] = {}
    attribute_candidates = {
        "color": _extract_phrase_matches(phrase, COLOR_PHRASES),
        "silhouette": _extract_phrase_matches(phrase, SILHOUETTE_PHRASES),
        "material": _extract_phrase_matches(phrase, MATERIAL_PHRASES),
        "pattern": _extract_phrase_matches(phrase, PATTERN_PHRASES),
        "texture": _extract_phrase_matches(phrase, TEXTURE_PHRASES),
        "style_tags": _extract_phrase_matches(phrase, STYLE_TAG_PHRASES),
    }

    for feature, values in attribute_candidates.items():
        current_values = list(getattr(item, feature))
        if current_values or not values:
            continue
        updates[feature] = values

    if not updates:
        return item
    return replace(item, **updates)


def _extract_phrase_matches(phrase: str, lexicon: tuple[str, ...]) -> list[str]:
    matches: list[str] = []
    padded_phrase = f" {phrase} "
    for candidate in lexicon:
        normalized_candidate = _normalize_value(candidate)
        if not normalized_candidate:
            continue
        if f" {normalized_candidate} " not in padded_phrase:
            continue
        if normalized_candidate not in matches:
            matches.append(normalized_candidate)
    return matches


def _dedupe(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_value(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _debug_log(label: str, payload: Any) -> None:
    if os.environ.get("LUXIA_PARSER_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return
    print(f"[v3_luxia_parser] {label}: {payload}", file=sys.stderr)
