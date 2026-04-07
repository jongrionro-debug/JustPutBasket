"""Luxia-backed structured query parser for the V3 pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import os
import re
import sys
from typing import Any

import requests

from .concepts import (
    STYLE_CONCEPT_CANONICALS,
    dedupe_preserve_order,
    extract_style_concepts,
    normalize_token,
)
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
    "style_concepts",
)
ATTRIBUTE_PRIORITY_NAMES = tuple(name for name in ITEM_ATTRIBUTE_NAMES if name != "category")
ITEM_STYLE_TAG_ALIASES = ("style_tags", "styles", "style")
ITEM_STYLE_CONCEPT_ALIASES = ("style_concepts", "concepts", "mood", "era")
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
MULTI_ITEM_CONNECTORS = (
    " with ",
    " and ",
    " & ",
    " plus ",
    " paired with ",
    " layered with ",
    " over ",
    " under ",
    ", ",
)
MULTI_ITEM_SPLIT_PATTERN = re.compile(
    r"\s(?:with|and|&|plus|paired with|layered with|over|under)\s|,\s*",
    re.IGNORECASE,
)
CATEGORY_ALIASES = {
    "pant": "pants",
    "pants": "pants",
    "trouser": "trousers",
    "trousers": "trousers",
    "jean": "jeans",
    "jeans": "jeans",
    "jacket": "jacket",
    "coat": "coat",
    "blazer": "blazer",
    "dress": "dress",
    "skirt": "skirt",
    "shirt": "shirt",
    "t shirt": "t shirt",
    "tee": "t shirt",
    "top": "top",
    "blouse": "blouse",
    "sweater": "sweater",
    "knit": "knit",
    "cardigan": "cardigan",
    "hoodie": "hoodie",
    "short": "shorts",
    "shorts": "shorts",
    "vest": "vest",
    "waistcoat": "vest",
    "jumpsuit": "jumpsuit",
    "trench": "trench",
}
CATEGORY_PHRASES = tuple(
    sorted(CATEGORY_ALIASES.keys(), key=lambda value: (-len(value), value))
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
                "pattern, texture, style_tags, style_concepts, required_attributes, "
                "preferred_attributes. "
                "Do not put item attributes at the top level. "
                "For multi-item queries, split attributes across the correct items. "
                "If a raw phrase explicitly says a color, silhouette, material, pattern, texture, or style "
                "for one item, copy that attribute into that item's structured fields. "
                "Put search concepts like vintage, minimal, romantic, retro, and avant-garde into "
                "style_concepts, not style_tags. "
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
    rule_first_targets = _build_rule_first_targets(query_text)
    try:
        target_items = _coerce_target_items(payload.get("target_items"), query_text=query_text)
    except RuntimeError:
        if not rule_first_targets:
            raise
        target_items = list(rule_first_targets)
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
    if rule_first_targets:
        parsed_query = _merge_rule_first_targets(parsed_query, rule_first_targets)
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
        concept_values = []
        for key in ITEM_STYLE_CONCEPT_ALIASES:
            concept_values.extend(_coerce_string_list(raw_item.get(key)))

        target_item = V3TargetItem(
            target_item_id=target_item_id,
            category=category,
            color=_coerce_string_list(raw_item.get("color")),
            silhouette=_coerce_string_list(raw_item.get("silhouette")),
            material=_coerce_string_list(raw_item.get("material")),
            pattern=_coerce_string_list(raw_item.get("pattern")),
            texture=_coerce_string_list(raw_item.get("texture")),
            style_tags=_dedupe(style_values),
            style_concepts=_coerce_style_concept_list(concept_values),
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


def _coerce_style_concept_list(values: list[str]) -> list[str]:
    inferred: list[str] = []
    for value in values:
        extracted = extract_style_concepts(value)
        if extracted:
            inferred.extend(extracted)
            continue
        normalized = _normalize_value(value)
        if normalized in STYLE_CONCEPT_CANONICALS:
            inferred.append(normalized)
    return dedupe_preserve_order(inferred)


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
                    repaired_style_preferences.pop(feature, None)
                    repaired_global_constraints.pop(feature, None)
                    continue

                migrated_values = repaired_style_preferences.pop(feature, None)
                if not migrated_values:
                    migrated_values = repaired_global_constraints.pop(feature, None)
                if not migrated_values:
                    continue

                repaired_item = replace(repaired_item, **{feature: migrated_values})

        repaired_item = _repair_item_attributes_from_raw_phrase(repaired_item)
        repaired_item = _repair_item_category_from_phrase(
            repaired_item,
            fallback_phrase=parsed_query.query_text if single_item_query else "",
        )
        repaired_item = _promote_explicit_attribute_priority(repaired_item)
        repaired_item = _promote_explicit_style_priority(repaired_item)
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
        "style_concepts": extract_style_concepts(phrase),
    }

    for feature, values in attribute_candidates.items():
        current_values = list(getattr(item, feature))
        if current_values or not values:
            continue
        updates[feature] = values

    if not updates:
        return item
    return replace(item, **updates)


def _build_rule_first_targets(query_text: str) -> list[V3TargetItem]:
    normalized_query = _normalize_value(query_text)
    if not normalized_query:
        return []

    raw_phrases = _split_multi_item_query(normalized_query)
    if not raw_phrases:
        raw_phrases = [normalized_query]

    targets: list[V3TargetItem] = []
    for index, phrase in enumerate(raw_phrases, start=1):
        category = _infer_category_from_phrase(phrase)
        if not category:
            return []

        color = _extract_phrase_matches(phrase, COLOR_PHRASES)
        silhouette = _extract_phrase_matches(phrase, SILHOUETTE_PHRASES)
        material = _extract_phrase_matches(phrase, MATERIAL_PHRASES)
        pattern = _extract_phrase_matches(phrase, PATTERN_PHRASES)
        texture = _extract_phrase_matches(phrase, TEXTURE_PHRASES)
        style_concepts = extract_style_concepts(phrase)

        required_attributes: list[str] = []
        preferred_attributes: list[str] = []
        for feature, values in (
            ("color", color),
            ("silhouette", silhouette),
            ("material", material),
            ("pattern", pattern),
            ("texture", texture),
        ):
            if values:
                required_attributes.append(feature)
        if style_concepts:
            preferred_attributes.append("style_concepts")

        targets.append(
            V3TargetItem(
                target_item_id=f"item_{index}",
                category=category,
                color=color,
                silhouette=silhouette,
                material=material,
                pattern=pattern,
                texture=texture,
                style_concepts=style_concepts,
                required_attributes=required_attributes,
                preferred_attributes=preferred_attributes,
                raw_phrase=phrase,
            )
        )
    return targets


def _split_multi_item_query(query_text: str) -> list[str]:
    if not any(connector in f" {query_text} " for connector in MULTI_ITEM_CONNECTORS):
        return []
    parts = [
        _normalize_value(part)
        for part in MULTI_ITEM_SPLIT_PATTERN.split(query_text)
    ]
    return [part for part in parts if part]


def _merge_rule_first_targets(
    parsed_query: V3ParsedQuery,
    rule_first_targets: list[V3TargetItem],
) -> V3ParsedQuery:
    if not rule_first_targets:
        return parsed_query
    if len(parsed_query.target_items) != len(rule_first_targets):
        return replace(parsed_query, target_items=list(rule_first_targets))

    merged_items: list[V3TargetItem] = []
    for item, rule_first_item in zip(parsed_query.target_items, rule_first_targets, strict=True):
        merged_required = _dedupe(item.required_attributes + rule_first_item.required_attributes)
        merged_items.append(
            replace(
                item,
                target_item_id=item.target_item_id or rule_first_item.target_item_id,
                category=rule_first_item.category or item.category,
                color=rule_first_item.color or item.color,
                silhouette=rule_first_item.silhouette or item.silhouette,
                material=rule_first_item.material or item.material,
                pattern=rule_first_item.pattern or item.pattern,
                texture=rule_first_item.texture or item.texture,
                style_concepts=rule_first_item.style_concepts or item.style_concepts,
                required_attributes=merged_required,
                preferred_attributes=[
                    feature
                    for feature in _dedupe(item.preferred_attributes + rule_first_item.preferred_attributes)
                    if feature not in set(merged_required)
                ],
                raw_phrase=item.raw_phrase or rule_first_item.raw_phrase,
            )
        )
    return replace(parsed_query, target_items=merged_items)


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


def _infer_category_from_phrase(phrase: str) -> str:
    padded_phrase = f" {phrase} "
    for candidate in CATEGORY_PHRASES:
        normalized_candidate = _normalize_value(candidate)
        if f" {normalized_candidate} " not in padded_phrase:
            continue
        return CATEGORY_ALIASES[candidate]
    return ""


def _repair_item_category_from_phrase(
    item: V3TargetItem,
    *,
    fallback_phrase: str = "",
) -> V3TargetItem:
    phrase = _normalize_value(item.raw_phrase or fallback_phrase)
    inferred_category = _infer_category_from_phrase(phrase)
    if not inferred_category:
        return item
    if item.category == inferred_category:
        return item
    return replace(item, category=inferred_category)


def _promote_explicit_attribute_priority(item: V3TargetItem) -> V3TargetItem:
    phrase = _normalize_value(item.raw_phrase)
    if not phrase:
        return item

    required_attributes = list(item.required_attributes)
    preferred_attributes = list(item.preferred_attributes)
    for feature in ("color", "silhouette", "material", "pattern", "texture"):
        values = list(getattr(item, feature))
        explicit_values = [value for value in values if f" {value} " in f" {phrase} "]
        if not explicit_values:
            continue
        if feature not in required_attributes:
            required_attributes.append(feature)
        preferred_attributes = [value for value in preferred_attributes if value != feature]

    return replace(
        item,
        required_attributes=required_attributes,
        preferred_attributes=preferred_attributes,
    )


def _promote_explicit_style_priority(item: V3TargetItem) -> V3TargetItem:
    phrase = _normalize_value(item.raw_phrase)
    if not phrase or not item.style_concepts:
        return item

    explicit_concepts = [
        style_concept
        for style_concept in item.style_concepts
        if f" {style_concept} " in f" {phrase} "
    ]
    if not explicit_concepts:
        return item

    required_attributes = list(item.required_attributes)
    preferred_attributes = list(item.preferred_attributes)
    if "style_concepts" not in required_attributes and "style_concepts" not in preferred_attributes:
        preferred_attributes.append("style_concepts")

    return replace(
        item,
        required_attributes=required_attributes,
        preferred_attributes=preferred_attributes,
    )


def _dedupe(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if value and value not in normalized:
            normalized.append(value)
    return normalized


def _normalize_value(value: str) -> str:
    return normalize_token(value)


def _debug_log(label: str, payload: Any) -> None:
    if os.environ.get("LUXIA_PARSER_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return
    print(f"[v3_luxia_parser] {label}: {payload}", file=sys.stderr)
