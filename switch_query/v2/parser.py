"""Deterministic query parsing for the V2 text/tag retrieval pipeline."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .documents import compose_query_document_text
from .models import ATTRIBUTE_NAMES, MULTI_VALUE_ATTRIBUTES, MULTI_VALUE_SEPARATOR, V2ParsedQuery


@dataclass(slots=True)
class _FeatureMatch:
    canonical: str
    raw_phrase: str
    start: int
    end: int


class V2QueryParser:
    def __init__(self, feature_vocabulary: dict[str, dict[str, str]] | None = None) -> None:
        self.feature_vocabulary = feature_vocabulary or {
            feature: {} for feature in ATTRIBUTE_NAMES
        }

    def parse(
        self,
        query_text: str,
        *,
        stage: str,
        balance_score: float,
        user_uploaded_image: str | None = None,
    ) -> V2ParsedQuery:
        del stage, balance_score, user_uploaded_image

        search_text = _normalize_search_text(query_text)
        canonical_tags: dict[str, str] = {}
        raw_phrases: dict[str, str] = {}

        for feature in ATTRIBUTE_NAMES:
            matches = self._find_matches(
                feature=feature,
                search_text=search_text,
                allow_multiple=feature in MULTI_VALUE_ATTRIBUTES,
            )
            if not matches:
                continue

            canonical_values: list[str] = []
            for match in matches:
                if match.canonical not in canonical_values:
                    canonical_values.append(match.canonical)
            canonical_tags[feature] = MULTI_VALUE_SEPARATOR.join(canonical_values)

            first_start = min(match.start for match in matches)
            last_end = max(match.end for match in matches)
            raw_phrases[feature] = search_text[first_start:last_end].strip()

        query_document = compose_query_document_text(
            query_text=query_text,
            canonical_tags=canonical_tags,
            raw_phrases=raw_phrases,
        )
        return V2ParsedQuery(
            query_text=query_text,
            canonical_tags=canonical_tags,
            raw_phrases=raw_phrases,
            query_document=query_document,
        )

    def _find_matches(
        self,
        *,
        feature: str,
        search_text: str,
        allow_multiple: bool,
    ) -> list[_FeatureMatch]:
        variant_map = self.feature_vocabulary.get(feature, {})
        if not variant_map:
            return []

        candidates = sorted(variant_map.items(), key=lambda item: (-len(item[0]), item[0]))
        matches: list[_FeatureMatch] = []
        occupied: list[tuple[int, int]] = []
        for variant, canonical in candidates:
            if not variant:
                continue
            pattern = re.compile(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])")
            found = pattern.search(search_text)
            if found is None:
                continue
            span = (found.start(), found.end())
            if any(_overlaps(span, used_span) for used_span in occupied):
                continue
            matches.append(
                _FeatureMatch(
                    canonical=canonical,
                    raw_phrase=search_text[found.start() : found.end()].strip(),
                    start=found.start(),
                    end=found.end(),
                )
            )
            occupied.append(span)
            if not allow_multiple:
                break
        matches.sort(key=lambda match: match.start)
        return matches


def _normalize_search_text(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]
