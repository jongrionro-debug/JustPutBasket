"""Thin query validation helpers for the V3 parser contract."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .models import V3ParsedQuery, V3TargetItem

ITEM_ATTRIBUTE_KEYS = frozenset(
    {
        "category",
        "color",
        "silhouette",
        "material",
        "pattern",
        "texture",
        "style_tags",
    }
)


@dataclass(slots=True)
class V3QueryValidationResult:
    validated_query: V3ParsedQuery
    violations: list[str]
    repair_notes: list[str]


def ensure_valid_v3_parsed_query(
    parsed_query: V3ParsedQuery,
    *,
    payload: dict[str, Any] | None = None,
) -> V3QueryValidationResult:
    result = validate_v3_parsed_query(parsed_query, payload=payload)
    if result.violations:
        joined = ", ".join(result.violations)
        raise RuntimeError(f"Invalid V3 parsed query: {joined}")
    return result


def validate_v3_parsed_query(
    parsed_query: V3ParsedQuery,
    *,
    payload: dict[str, Any] | None = None,
) -> V3QueryValidationResult:
    violations: list[str] = []
    repair_notes: list[str] = []
    validated_query = parsed_query

    if payload:
        orphan_keys = ITEM_ATTRIBUTE_KEYS.intersection(payload.keys())
        if orphan_keys:
            violations.append("orphan_global_attribute")
            repair_notes.append(
                "top-level item attributes are not allowed: " + ",".join(sorted(orphan_keys))
            )

    if not validated_query.target_items:
        violations.append("empty_target_items")
        return V3QueryValidationResult(
            validated_query=validated_query,
            violations=_dedupe(violations),
            repair_notes=repair_notes,
        )

    normalized_items: list[V3TargetItem] = []
    multi_item_query = len(validated_query.target_items) > 1

    for item in validated_query.target_items:
        if not item.category.strip():
            violations.append("missing_target_category")

        normalized_item = replace(
            item,
            required_attributes=_dedupe(item.required_attributes),
            preferred_attributes=[
                attribute
                for attribute in _dedupe(item.preferred_attributes)
                if attribute not in set(item.required_attributes)
            ],
        )

        if multi_item_query and not normalized_item.raw_phrase.strip():
            violations.append("ambiguous_attribute_binding")

        normalized_items.append(normalized_item)

    validated_query = replace(validated_query, target_items=normalized_items)

    if multi_item_query:
        for index, item in enumerate(validated_query.target_items):
            phrase = _normalize_text(item.raw_phrase)
            if item.category and not _contains_token(phrase, item.category):
                category_positions = _token_positions(validated_query.target_items, item.category)
                if any(position != index for position in category_positions):
                    violations.append("ambiguous_attribute_binding")

            for color in item.color:
                if _contains_token(phrase, color):
                    continue
                positions = _token_positions(validated_query.target_items, color)
                if any(position != index for position in positions):
                    violations.append("cross_item_attribute_swap")

    return V3QueryValidationResult(
        validated_query=validated_query,
        violations=_dedupe(violations),
        repair_notes=repair_notes,
    )


def _token_positions(items: list[V3TargetItem], token: str) -> set[int]:
    positions: set[int] = set()
    for index, item in enumerate(items):
        if _contains_token(_normalize_text(item.raw_phrase), token):
            positions.add(index)
    return positions


def _contains_token(text: str, token: str) -> bool:
    normalized_token = _normalize_text(token)
    if not normalized_token:
        return False
    return normalized_token in text


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _dedupe(values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in values:
        if value and value not in normalized:
            normalized.append(value)
    return normalized
