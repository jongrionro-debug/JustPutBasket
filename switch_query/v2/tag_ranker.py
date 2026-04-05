"""Tag-based ranking for the V2 retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .explanation import build_rank_explanation
from .models import (
    ATTRIBUTE_NAMES,
    MULTI_VALUE_ATTRIBUTES,
    MULTI_VALUE_SEPARATOR,
    V2ArchiveDocument,
    V2ParsedQuery,
    V2RankedResult,
)

FeatureMatchStatus = Literal["exact", "partial", "missing", "contradiction"]
DETAIL_COLOR_CONTRADICTION_PENALTY = -8.0
DETAIL_COLOR_LEXICON = {
    "black",
    "white",
    "beige",
    "brown",
    "tan",
    "camel",
    "gray",
    "grey",
    "silver",
    "charcoal",
    "navy",
    "blue",
    "red",
    "burgundy",
    "maroon",
    "pink",
    "green",
    "olive",
    "khaki",
    "yellow",
    "orange",
    "purple",
    "lavender",
    "gold",
    "cream",
    "ivory",
}


@dataclass(slots=True)
class FeatureMatchResult:
    feature: str
    status: FeatureMatchStatus
    overlap_values: list[str] = field(default_factory=list)
    query_values: list[str] = field(default_factory=list)
    candidate_values: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DetailColorSignal:
    confirmed_phrase: str | None = None
    contradiction_phrase: str | None = None


@dataclass(slots=True)
class TagRankerConfig:
    top_k: int = 20
    category_hard_filter: bool = True


@dataclass(slots=True)
class TagRanker:
    config: TagRankerConfig = field(default_factory=TagRankerConfig)

    def rank(
        self,
        parsed_query: V2ParsedQuery,
        documents: Sequence[V2ArchiveDocument],
    ) -> list[V2RankedResult]:
        scored: list[V2RankedResult] = []
        for document in documents:
            if self.config.category_hard_filter and _fails_category_hard_filter(parsed_query, document):
                continue
            scored.append(_score_document(parsed_query, document))

        scored.sort(key=lambda result: (-result.score, result.image_id))
        return scored[: self.config.top_k]


def classify_feature_match(
    feature: str,
    query_value: str,
    candidate_value: str,
) -> FeatureMatchResult:
    query_values = _split_values(query_value)
    candidate_values = _split_values(candidate_value)

    if not candidate_values:
        return FeatureMatchResult(
            feature=feature,
            status="missing",
            query_values=query_values,
            candidate_values=candidate_values,
        )

    if feature in MULTI_VALUE_ATTRIBUTES:
        overlap_values = [value for value in query_values if value in set(candidate_values)]
        if query_values == candidate_values:
            return FeatureMatchResult(
                feature=feature,
                status="exact",
                overlap_values=overlap_values,
                query_values=query_values,
                candidate_values=candidate_values,
            )
        if overlap_values:
            return FeatureMatchResult(
                feature=feature,
                status="partial",
                overlap_values=overlap_values,
                query_values=query_values,
                candidate_values=candidate_values,
            )
        return FeatureMatchResult(
            feature=feature,
            status="contradiction",
            query_values=query_values,
            candidate_values=candidate_values,
        )

    if query_values == candidate_values:
        return FeatureMatchResult(
            feature=feature,
            status="exact",
            overlap_values=query_values,
            query_values=query_values,
            candidate_values=candidate_values,
        )

    return FeatureMatchResult(
        feature=feature,
        status="contradiction",
        query_values=query_values,
        candidate_values=candidate_values,
    )


def _score_document(parsed_query: V2ParsedQuery, document: V2ArchiveDocument) -> V2RankedResult:
    score = 0.0
    matched_attributes: dict[str, str] = {}
    mismatched_attributes: dict[str, str] = {}
    missing_attributes: dict[str, str] = {}
    score_breakdown: dict[str, float] = {}
    match_reasons: list[str] = []
    category_match = classify_feature_match(
        "category",
        parsed_query.canonical_tags.get("category", "").strip(),
        document.canonical_tags.get("category", "").strip(),
    )
    detail_color_signal = _evaluate_detail_color_signal(parsed_query, document)

    for feature in ATTRIBUTE_NAMES:
        query_value = parsed_query.canonical_tags.get(feature, "").strip()
        if not query_value:
            continue

        candidate_value = document.canonical_tags.get(feature, "").strip()
        match = classify_feature_match(feature, query_value, candidate_value)

        if match.status == "exact":
            matched_value = MULTI_VALUE_SEPARATOR.join(match.overlap_values)
            matched_attributes[feature] = matched_value
            feature_score = _score_for_match(
                feature,
                match.status,
                category_match=category_match,
                detail_color_signal=detail_color_signal,
            )
            if feature_score:
                score += feature_score
                score_breakdown[f"{feature}:exact"] = feature_score
            match_reasons.append(f"{feature} exact match")
            continue

        if match.status == "partial":
            matched_value = MULTI_VALUE_SEPARATOR.join(match.overlap_values)
            matched_attributes[feature] = matched_value
            feature_score = _score_for_match(
                feature,
                match.status,
                category_match=category_match,
                detail_color_signal=detail_color_signal,
            )
            if feature_score:
                score += feature_score
                score_breakdown[f"{feature}:partial"] = feature_score
            match_reasons.append(f"{feature} partial match")
            continue

        if match.status == "missing":
            missing_attributes[feature] = query_value
            penalty = _missing_penalty(feature, parsed_query)
            if penalty:
                score += penalty
                score_breakdown[f"{feature}:missing"] = penalty
            match_reasons.append(f"{feature} missing")
            continue

        mismatched_attributes[feature] = candidate_value
        penalty = _contradiction_penalty(feature)
        if penalty:
            score += penalty
            score_breakdown[f"{feature}:contradiction"] = penalty
        match_reasons.append(f"{feature} contradiction")

    if detail_color_signal.confirmed_phrase is not None:
        matched_attributes["detail_color_confirmation"] = detail_color_signal.confirmed_phrase
        score_breakdown["color:detail_target_confirmation"] = 2.0
        score += 2.0
        match_reasons.append("detail target color confirmation")

    if detail_color_signal.contradiction_phrase is not None:
        mismatched_attributes["detail"] = detail_color_signal.contradiction_phrase
        score += DETAIL_COLOR_CONTRADICTION_PENALTY
        score_breakdown["detail:target_color_contradiction"] = DETAIL_COLOR_CONTRADICTION_PENALTY
        match_reasons.append("detail target color contradiction")

    explanation = build_rank_explanation(
        parsed_query,
        matched_attributes=matched_attributes,
        mismatched_attributes=mismatched_attributes,
        missing_attributes=missing_attributes,
        score_breakdown=score_breakdown,
    )
    return V2RankedResult(
        image_id=document.image_id,
        file_path=document.file_path,
        brand=document.brand,
        season_group=document.season_group,
        score=round(score, 6),
        matched_attributes=matched_attributes,
        mismatched_attributes=mismatched_attributes,
        missing_attributes=missing_attributes,
        score_breakdown=score_breakdown,
        match_reasons=match_reasons,
        explanation=explanation,
    )


def _fails_category_hard_filter(parsed_query: V2ParsedQuery, document: V2ArchiveDocument) -> bool:
    query_value = parsed_query.canonical_tags.get("category", "").strip()
    if not query_value:
        return False
    match = classify_feature_match("category", query_value, document.canonical_tags.get("category", ""))
    return match.status in {"missing", "contradiction"}


def _score_for_match(
    feature: str,
    status: FeatureMatchStatus,
    *,
    category_match: FeatureMatchResult,
    detail_color_signal: DetailColorSignal,
) -> float:
    if feature == "category":
        return 8.0 if status == "exact" else 2.0 if status == "partial" else 0.0
    if feature == "color":
        return _score_for_color_exact(status, category_match, detail_color_signal)
    if feature == "silhouette":
        return 4.0 if status == "exact" else 2.0 if status == "partial" else 0.0
    if feature == "mood":
        return 3.0 if status == "exact" else 1.0 if status == "partial" else 0.0
    if feature in {"material", "pattern", "texture", "era"}:
        return 2.0 if status == "exact" else 0.0
    if feature == "detail":
        return 1.0 if status in {"exact", "partial"} else 0.0
    return 0.0


def _missing_penalty(feature: str, parsed_query: V2ParsedQuery) -> float:
    if feature in set(parsed_query.required_features):
        return -10.0
    return 0.0


def _contradiction_penalty(feature: str) -> float:
    if feature == "category":
        return -12.0
    if feature == "color":
        return -8.0
    return 0.0


def _split_values(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(MULTI_VALUE_SEPARATOR) if item.strip()]


def _score_for_color_exact(
    status: FeatureMatchStatus,
    category_match: FeatureMatchResult,
    detail_color_signal: DetailColorSignal,
) -> float:
    if status != "exact":
        return 0.0
    if category_match.status == "exact":
        return 6.0
    if category_match.status == "partial":
        return 4.0 if detail_color_signal.confirmed_phrase is not None else 2.0
    return 0.0


def _evaluate_detail_color_signal(
    parsed_query: V2ParsedQuery,
    document: V2ArchiveDocument,
) -> DetailColorSignal:
    query_categories = _split_values(parsed_query.canonical_tags.get("category", ""))
    query_colors = _split_values(parsed_query.canonical_tags.get("color", ""))
    detail_phrases = _split_values(document.canonical_tags.get("detail", ""))
    if not query_categories or not query_colors or not detail_phrases:
        return DetailColorSignal()

    normalized_query_categories = {_normalize_fragment(value) for value in query_categories}
    normalized_query_colors = {_normalize_fragment(value) for value in query_colors}

    for phrase in detail_phrases:
        normalized_phrase = _normalize_fragment(phrase)
        category = _find_matching_category(normalized_phrase, normalized_query_categories)
        if category is None:
            continue
        phrase_colors = _extract_pre_category_colors(normalized_phrase, category)
        if not phrase_colors:
            continue
        if phrase_colors & normalized_query_colors:
            if phrase_colors <= normalized_query_colors:
                return DetailColorSignal(confirmed_phrase=phrase)
            conflicting_colors = sorted(phrase_colors - normalized_query_colors)
            return DetailColorSignal(
                contradiction_phrase=f"{phrase} ({'/'.join(conflicting_colors)})"
            )
        conflicting_colors = sorted(phrase_colors)
        return DetailColorSignal(contradiction_phrase=f"{phrase} ({'/'.join(conflicting_colors)})")
    return DetailColorSignal()


def _find_matching_category(phrase: str, categories: set[str]) -> str | None:
    for category in sorted(categories, key=len, reverse=True):
        if category in phrase:
            return category
    return None


def _extract_pre_category_colors(phrase: str, category: str) -> set[str]:
    prefix = phrase.split(category, 1)[0]
    words = set(prefix.split())
    return {color for color in DETAIL_COLOR_LEXICON if color in words}


def _normalize_fragment(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())
