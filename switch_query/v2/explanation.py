"""Rule-based explanation builders for the V2 text/tag retrieval pipeline."""

from __future__ import annotations

from .models import ATTRIBUTE_NAMES, MULTI_VALUE_SEPARATOR


def explain_match(
    query_tags: dict[str, str],
    candidate_tags: dict[str, str],
) -> tuple[dict[str, str], dict[str, str], dict[str, str], str]:
    matched: dict[str, str] = {}
    mismatched: dict[str, str] = {}
    missing: dict[str, str] = {}

    for feature in ATTRIBUTE_NAMES:
        query_value = query_tags.get(feature, "").strip()
        if not query_value:
            continue

        candidate_value = candidate_tags.get(feature, "").strip()
        if not candidate_value:
            missing[feature] = query_value
            continue

        query_values = _split_values(query_value)
        candidate_values = _split_values(candidate_value)
        overlap = [value for value in query_values if value in set(candidate_values)]
        if overlap:
            matched[feature] = MULTI_VALUE_SEPARATOR.join(overlap)
        if set(query_values) != set(candidate_values):
            mismatched[feature] = candidate_value

    return matched, mismatched, missing, _serialize_explanation(matched, mismatched, missing)


def _split_values(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(MULTI_VALUE_SEPARATOR) if item.strip()]


def _serialize_explanation(
    matched: dict[str, str],
    mismatched: dict[str, str],
    missing: dict[str, str],
) -> str:
    return " | ".join(
        [
            _format_section("matched", matched),
            _format_section("mismatched", mismatched),
            _format_section("missing", missing),
        ]
    )


def _format_section(label: str, payload: dict[str, str]) -> str:
    if not payload:
        return f"{label}: none"
    values = ", ".join(f"{feature}={value}" for feature, value in payload.items())
    return f"{label}: {values}"
