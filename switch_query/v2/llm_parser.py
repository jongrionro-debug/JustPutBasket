"""Luxia-backed structured query parser for the V2 retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import sys
from typing import Any

import requests

from .documents import compose_query_document_text
from .models import ATTRIBUTE_NAMES, MULTI_VALUE_SEPARATOR, V2ParsedQuery

LUXIA_CHAT_URL = "https://bridge.luxiacloud.com/luxia/v1/chat"


@dataclass(slots=True)
class LuxiaQueryParserConfig:
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
class LuxiaQueryParser:
    feature_vocabulary: dict[str, dict[str, str]] | None = None
    config: LuxiaQueryParserConfig = field(default_factory=LuxiaQueryParserConfig)
    session: requests.Session | None = None

    def parse(
        self,
        query_text: str,
        *,
        stage: str,
        balance_score: float,
        user_uploaded_image: str | None = None,
    ) -> V2ParsedQuery:
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
                return _build_parsed_query(query_text, structured_payload)
            except Exception as exc:
                last_error = exc
                _debug_log("parse_error", str(exc))

        if last_error is not None:
            raise RuntimeError(f"Luxia query parsing failed: {last_error}") from last_error
        raise RuntimeError("Luxia query parsing failed without a captured exception")

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
                    "messages": _build_messages(query_text, stage=stage, balance_score=balance_score),
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
                "Return only JSON with these keys: canonical_tags, raw_phrases, "
                "required_features, preferred_features, confidence. "
                "Use only these feature names: "
                f"{', '.join(ATTRIBUTE_NAMES)}. "
                f"Use '{MULTI_VALUE_SEPARATOR}' to join multiple values within a single field. "
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


def _build_parsed_query(query_text: str, payload: dict[str, Any]) -> V2ParsedQuery:
    required_payload = payload.get("required_features")
    preferred_payload = payload.get("preferred_features")

    canonical_tags = _coerce_feature_map(payload.get("canonical_tags"))
    if not canonical_tags:
        canonical_tags = _coerce_top_level_feature_map(payload)
    if not canonical_tags:
        canonical_tags = _coerce_feature_map(required_payload)
    if not canonical_tags:
        canonical_tags = _coerce_feature_map(preferred_payload)

    raw_phrases = _coerce_feature_map(payload.get("raw_phrases"))

    required_features = _coerce_feature_list(required_payload)
    preferred_features = _coerce_feature_list(preferred_payload)

    # Keep feature priorities aligned with extracted values only.
    required_features = [feature for feature in required_features if feature in canonical_tags]
    preferred_features = [
        feature
        for feature in preferred_features
        if feature in canonical_tags and feature not in required_features
    ]
    if not required_features and not preferred_features:
        required_features, preferred_features = _default_feature_priority(canonical_tags)

    confidence = _coerce_confidence(payload.get("confidence", 0.0))
    query_document = compose_query_document_text(
        query_text=query_text,
        canonical_tags=canonical_tags,
        raw_phrases=raw_phrases,
    )
    return V2ParsedQuery(
        query_text=query_text,
        canonical_tags=canonical_tags,
        raw_phrases=raw_phrases,
        required_features=required_features,
        preferred_features=preferred_features,
        confidence=confidence,
        query_document=query_document,
    )


def _coerce_feature_map(value: Any) -> dict[str, str]:
    if value is None:
        return {}

    if isinstance(value, list):
        return _coerce_feature_map_from_list(value)

    if isinstance(value, dict):
        feature_priority_map = _coerce_feature_map_from_priority_dict(value)
        if feature_priority_map:
            return feature_priority_map

    if not isinstance(value, dict):
        raise RuntimeError("Luxia parser output must contain object feature maps.")

    normalized: dict[str, str] = {}
    for feature, raw in value.items():
        if feature not in ATTRIBUTE_NAMES:
            continue
        values = _coerce_string_list(raw)
        if not values:
            continue
        normalized[feature] = MULTI_VALUE_SEPARATOR.join(values)
    return normalized


def _coerce_feature_map_from_priority_dict(value: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for feature, raw in value.items():
        if feature not in ATTRIBUTE_NAMES:
            continue
        if isinstance(raw, bool):
            continue
        try:
            values = _coerce_string_list(raw)
        except RuntimeError:
            continue
        if not values:
            continue
        normalized[feature] = MULTI_VALUE_SEPARATOR.join(values)
    return normalized


def _coerce_feature_map_from_list(value: list[Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        feature = str(item.get("feature", "")).strip()
        if feature not in ATTRIBUTE_NAMES:
            continue
        raw_value = item.get("values", item.get("value"))
        values = _coerce_string_list(raw_value)
        if not values:
            continue
        normalized[feature] = MULTI_VALUE_SEPARATOR.join(values)
    return normalized


def _coerce_top_level_feature_map(payload: dict[str, Any]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for feature in ATTRIBUTE_NAMES:
        if feature not in payload:
            continue
        try:
            values = _coerce_string_list(payload.get(feature))
        except RuntimeError:
            continue
        if not values:
            continue
        normalized[feature] = MULTI_VALUE_SEPARATOR.join(values)
    return normalized


def _coerce_feature_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.startswith("{") and normalized.endswith("}"):
            try:
                value = json.loads(normalized)
            except json.JSONDecodeError:
                value = [
                    part.strip()
                    for part in value.replace(",", MULTI_VALUE_SEPARATOR).split(MULTI_VALUE_SEPARATOR)
                ]
        else:
            value = [
                part.strip()
                for part in value.replace(",", MULTI_VALUE_SEPARATOR).split(MULTI_VALUE_SEPARATOR)
            ]
    if isinstance(value, dict):
        return _coerce_feature_list_from_dict(value)
    if not isinstance(value, list):
        raise RuntimeError("Luxia parser output must contain feature lists.")

    features: list[str] = []
    for item in value:
        if isinstance(item, dict):
            feature = _extract_feature_name(item)
        elif isinstance(item, str):
            feature = item.strip()
        else:
            continue
        if feature in ATTRIBUTE_NAMES and feature not in features:
            features.append(feature)
    return features


def _coerce_feature_list_from_dict(value: dict[str, Any]) -> list[str]:
    nested_items = value.get("features") or value.get("items") or value.get("values")
    if isinstance(nested_items, list):
        return _coerce_feature_list(nested_items)
    if isinstance(nested_items, str):
        return _coerce_feature_list(nested_items)

    features: list[str] = []
    for key, raw in value.items():
        if key in ATTRIBUTE_NAMES and bool(raw) and key not in features:
            features.append(key)
            continue
        if key not in {"feature", "name", "key"}:
            continue
        if not isinstance(raw, str):
            continue
        feature = raw.strip()
        if feature in ATTRIBUTE_NAMES and feature not in features:
            features.append(feature)
    return features


def _extract_feature_name(item: dict[str, Any]) -> str:
    for key in ("feature", "name", "key", "id"):
        raw = item.get(key)
        if isinstance(raw, str):
            return raw.strip()
    return ""


def _coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [part.strip() for part in value.split(MULTI_VALUE_SEPARATOR)]
    elif isinstance(value, list):
        items = [str(part).strip() for part in value]
    else:
        raise RuntimeError("Luxia parser feature values must be strings or string arrays.")

    cleaned: list[str] = []
    for item in items:
        normalized = _normalize_value(item)
        if normalized and normalized not in cleaned:
            cleaned.append(normalized)
    return cleaned


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


def _default_feature_priority(canonical_tags: dict[str, str]) -> tuple[list[str], list[str]]:
    required_features: list[str] = []
    preferred_features: list[str] = []
    for feature in canonical_tags:
        if feature in {"category", "color"}:
            required_features.append(feature)
        else:
            preferred_features.append(feature)
    return required_features, preferred_features


def _normalize_value(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _debug_log(label: str, payload: Any) -> None:
    if os.environ.get("LUXIA_PARSER_DEBUG", "").lower() not in {"1", "true", "yes", "on"}:
        return
    print(f"[luxia_parser] {label}: {payload}", file=sys.stderr)
