"""Symbolic item-aware ranking for the V3 retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product
from typing import Iterable, Sequence

from .models import (
    V3ArchiveDocument,
    V3DocumentItem,
    V3ItemAssignment,
    V3ParsedQuery,
    V3RankedResult,
    V3TargetItem,
)

ATTRIBUTE_KEYS = ("color", "silhouette", "material", "pattern", "texture", "style_tags")
ITEM_LEVEL_QUERY_FEATURES = frozenset({"category", *ATTRIBUTE_KEYS})
STYLE_PREFERENCE_ALLOWED_KEYS = frozenset({"mood", "era", "style_tags"})
MULTI_VALUE_SEPARATOR = "|"
COLOR_LEXICON = {
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
EXACT_WEIGHTS = {
    "color": 6.0,
    "silhouette": 4.0,
    "material": 2.0,
    "pattern": 2.0,
    "texture": 2.0,
    "style_tags": 2.0,
}
PARTIAL_WEIGHTS = {
    "color": 2.0,
    "silhouette": 2.0,
    "material": 1.0,
    "pattern": 1.0,
    "texture": 1.0,
    "style_tags": 1.0,
}
CONTRADICTION_WEIGHTS = {
    "color": -8.0,
    "silhouette": -4.0,
    "material": -2.0,
    "pattern": -2.0,
    "texture": -2.0,
    "style_tags": -1.5,
}
REQUIRED_EXACT_BONUS = 2.0
REQUIRED_PARTIAL_BONUS = 1.0
REQUIRED_MISSING_PENALTY = -6.0
REQUIRED_CONTRADICTION_PENALTY = -3.0
PREFERRED_EXACT_BONUS = 1.0
PREFERRED_PARTIAL_BONUS = 0.5
FALLBACK_SCALE = 0.45
FALLBACK_CONTRADICTION_SCALE = 0.25
FULL_COVERAGE_BONUS = 12.0
DETAIL_CONFIRMATION_BONUS = 1.5
DETAIL_CONTRADICTION_PENALTY = -1.5
GLOBAL_CONSTRAINT_EXACT_BONUS = 1.5
GLOBAL_CONSTRAINT_PARTIAL_BONUS = 0.75
STYLE_PREFERENCE_EXACT_BONUS = 1.0
STYLE_PREFERENCE_PARTIAL_BONUS = 0.5
MISSING_TARGET_PENALTY = -10.0
HARD_FAIL_PENALTY = -100.0


@dataclass(slots=True)
class V3RankerConfig:
    top_k: int = 20


@dataclass(slots=True)
class _MatchOutcome:
    status: str
    overlap_values: list[str] = field(default_factory=list)
    query_values: list[str] = field(default_factory=list)
    candidate_values: list[str] = field(default_factory=list)


@dataclass(slots=True)
class _CandidateOption:
    assignment: V3ItemAssignment
    score: float
    score_breakdown: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    used_item_id: str = ""


class V3Ranker:
    config: V3RankerConfig

    def __init__(self, config: V3RankerConfig | None = None) -> None:
        self.config = config or V3RankerConfig()

    def rank(
        self,
        parsed_query: V3ParsedQuery,
        documents: Sequence[V3ArchiveDocument],
    ) -> list[V3RankedResult]:
        scored = [_score_document(parsed_query, document) for document in documents]
        scored.sort(key=lambda result: (-result.score, result.image_id))
        return scored[: self.config.top_k]


def _score_document(parsed_query: V3ParsedQuery, document: V3ArchiveDocument) -> V3RankedResult:
    candidate_options = [
        _build_candidate_options(target_item, document)
        for target_item in parsed_query.target_items
    ]
    selected_options = _select_best_assignment(candidate_options)

    item_assignments = [option.assignment for option in selected_options]
    score_breakdown: dict[str, float] = {}
    match_reasons: list[str] = []
    score = 0.0

    for option in selected_options:
        score += option.score
        score_breakdown.update(option.score_breakdown)
        match_reasons.extend(option.reasons)
        score_breakdown[f"item:{option.assignment.target_item_id}:total"] = round(option.score, 6)

    matched_count = sum(
        assignment.status in {"exact", "partial", "fallback_match"}
        for assignment in item_assignments
    )
    contradiction_count = sum(
        assignment.status in {"contradiction", "hard_fail"}
        for assignment in item_assignments
    )
    if matched_count == len(parsed_query.target_items) and contradiction_count == 0:
        score += FULL_COVERAGE_BONUS
        score_breakdown["coverage:full_item_set"] = FULL_COVERAGE_BONUS
        match_reasons.append("full item set match")

    detail_score, detail_breakdown, detail_reasons = _score_detail_consistency(parsed_query, document)
    score += detail_score
    score_breakdown.update(detail_breakdown)
    match_reasons.extend(detail_reasons)

    auxiliary_score, auxiliary_breakdown, auxiliary_reasons = _score_auxiliary_constraints(
        parsed_query,
        document,
    )
    score += auxiliary_score
    score_breakdown.update(auxiliary_breakdown)
    match_reasons.extend(auxiliary_reasons)

    hard_fail_targets = _detect_cross_item_swap_targets(parsed_query, document)
    if hard_fail_targets:
        score += HARD_FAIL_PENALTY
        score_breakdown["penalty:cross_item_swap"] = HARD_FAIL_PENALTY
        match_reasons.append("cross-item attribute swap detected")
        item_assignments = [
            replace(
                assignment,
                status="hard_fail" if assignment.target_item_id in hard_fail_targets else assignment.status,
                notes=(
                    assignment.notes + ["cross-item attribute swap detected"]
                    if assignment.target_item_id in hard_fail_targets
                    else assignment.notes
                ),
            )
            for assignment in item_assignments
        ]

    explanation = _build_rank_explanation(item_assignments, score_breakdown)
    return V3RankedResult(
        image_id=document.image_id,
        file_path=document.file_path,
        brand=document.brand,
        season_group=document.season_group,
        score=round(score, 6),
        score_breakdown=score_breakdown,
        match_reasons=_dedupe(match_reasons),
        explanation=explanation,
        item_assignments=item_assignments,
    )


def _build_candidate_options(
    target_item: V3TargetItem,
    document: V3ArchiveDocument,
) -> list[_CandidateOption]:
    matching_items = [
        item for item in document.items if _normalize(item.category) == _normalize(target_item.category)
    ]
    options: list[_CandidateOption] = []

    for item in matching_items:
        options.append(_evaluate_item_candidate(target_item, item))

    if not matching_items:
        fallback_option = _evaluate_fallback_candidate(target_item, document)
        if fallback_option is not None:
            options.append(fallback_option)

    options.append(_build_missing_candidate(target_item))
    return options


def _evaluate_item_candidate(
    target_item: V3TargetItem,
    document_item: V3DocumentItem,
) -> _CandidateOption:
    assignment = V3ItemAssignment(
        target_item_id=target_item.target_item_id,
        target_category=target_item.category,
        status="exact",
        matched_item_id=document_item.item_id,
        matched_category=document_item.category,
        source="item",
    )
    breakdown: dict[str, float] = {}
    reasons = [f"{target_item.category} item candidate"]
    score = 8.0
    breakdown[f"item:{target_item.target_item_id}:category:exact"] = 8.0
    assignment.matched_attributes["category"] = document_item.category
    score += _priority_bonus(
        target_item,
        feature="category",
        exact=True,
        breakdown=breakdown,
    )

    saw_partial = False
    for feature in ATTRIBUTE_KEYS:
        query_values = list(getattr(target_item, feature))
        if not query_values:
            continue

        outcome = _classify_values(query_values, list(getattr(document_item, feature)))
        prefix = f"item:{target_item.target_item_id}:{feature}"
        if outcome.status == "exact":
            score += EXACT_WEIGHTS[feature]
            breakdown[f"{prefix}:exact"] = EXACT_WEIGHTS[feature]
            assignment.matched_attributes[feature] = MULTI_VALUE_SEPARATOR.join(outcome.overlap_values)
            reasons.append(f"{target_item.category} {feature} exact")
            score += _priority_bonus(
                target_item,
                feature=feature,
                exact=True,
                breakdown=breakdown,
            )
            continue

        if outcome.status == "partial":
            saw_partial = True
            score += PARTIAL_WEIGHTS[feature]
            breakdown[f"{prefix}:partial"] = PARTIAL_WEIGHTS[feature]
            assignment.matched_attributes[feature] = MULTI_VALUE_SEPARATOR.join(outcome.overlap_values)
            reasons.append(f"{target_item.category} {feature} partial")
            score += _priority_bonus(
                target_item,
                feature=feature,
                exact=False,
                breakdown=breakdown,
            )
            continue

        if outcome.status == "missing":
            if feature in set(target_item.required_attributes):
                score += REQUIRED_MISSING_PENALTY
                breakdown[f"{prefix}:missing_required"] = REQUIRED_MISSING_PENALTY
            assignment.missing_attributes.append(feature)
            saw_partial = True
            reasons.append(f"{target_item.category} {feature} missing")
            continue

        contradiction_penalty = CONTRADICTION_WEIGHTS[feature]
        score += contradiction_penalty
        breakdown[f"{prefix}:contradiction"] = contradiction_penalty
        assignment.contradicted_attributes[feature] = MULTI_VALUE_SEPARATOR.join(outcome.candidate_values)
        reasons.append(f"{target_item.category} {feature} contradiction")
        if feature in set(target_item.required_attributes):
            score += REQUIRED_CONTRADICTION_PENALTY
            breakdown[f"{prefix}:required_contradiction"] = REQUIRED_CONTRADICTION_PENALTY

    if assignment.contradicted_attributes:
        assignment.status = "contradiction"
    elif saw_partial:
        assignment.status = "partial"

    assignment.score = round(score, 6)
    return _CandidateOption(
        assignment=assignment,
        score=score,
        score_breakdown=breakdown,
        reasons=reasons,
        used_item_id=document_item.item_id,
    )


def _evaluate_fallback_candidate(
    target_item: V3TargetItem,
    document: V3ArchiveDocument,
) -> _CandidateOption | None:
    category_values = _split_values(document.canonical_tags.get("category", ""))
    if _normalize(target_item.category) not in {_normalize(value) for value in category_values}:
        return None

    assignment = V3ItemAssignment(
        target_item_id=target_item.target_item_id,
        target_category=target_item.category,
        status="fallback_match",
        matched_item_id="fallback:canonical_tags",
        matched_category=target_item.category,
        source="fallback",
        notes=["used outfit-level fallback"],
    )
    breakdown: dict[str, float] = {}
    reasons = [f"{target_item.category} fallback candidate"]
    score = round(8.0 * FALLBACK_SCALE, 6)
    breakdown[f"item:{target_item.target_item_id}:category:fallback"] = score
    assignment.matched_attributes["category"] = target_item.category
    category_bonus = _priority_bonus(
        target_item,
        feature="category",
        exact=True,
        breakdown=breakdown,
        scale=FALLBACK_SCALE,
        prefix=f"item:{target_item.target_item_id}:category",
    )
    score += category_bonus

    saw_positive = False
    for feature in ATTRIBUTE_KEYS:
        query_values = list(getattr(target_item, feature))
        if not query_values:
            continue

        outcome = _classify_values(query_values, _split_values(document.canonical_tags.get(feature, "")))
        prefix = f"item:{target_item.target_item_id}:{feature}"
        if outcome.status == "exact":
            saw_positive = True
            component = round(EXACT_WEIGHTS[feature] * FALLBACK_SCALE, 6)
            score += component
            breakdown[f"{prefix}:fallback_exact"] = component
            assignment.matched_attributes[feature] = MULTI_VALUE_SEPARATOR.join(outcome.overlap_values)
            reasons.append(f"{target_item.category} {feature} fallback exact")
            score += _priority_bonus(
                target_item,
                feature=feature,
                exact=True,
                breakdown=breakdown,
                scale=FALLBACK_SCALE,
                prefix=prefix,
            )
            continue

        if outcome.status == "partial":
            saw_positive = True
            component = round(PARTIAL_WEIGHTS[feature] * FALLBACK_SCALE, 6)
            score += component
            breakdown[f"{prefix}:fallback_partial"] = component
            assignment.matched_attributes[feature] = MULTI_VALUE_SEPARATOR.join(outcome.overlap_values)
            reasons.append(f"{target_item.category} {feature} fallback partial")
            score += _priority_bonus(
                target_item,
                feature=feature,
                exact=False,
                breakdown=breakdown,
                scale=FALLBACK_SCALE,
                prefix=prefix,
            )
            continue

        if outcome.status == "missing":
            if feature in set(target_item.required_attributes):
                component = round(REQUIRED_MISSING_PENALTY * FALLBACK_SCALE, 6)
                score += component
                breakdown[f"{prefix}:fallback_missing_required"] = component
            assignment.missing_attributes.append(feature)
            reasons.append(f"{target_item.category} {feature} fallback missing")
            continue

        component = round(CONTRADICTION_WEIGHTS[feature] * FALLBACK_CONTRADICTION_SCALE, 6)
        score += component
        breakdown[f"{prefix}:fallback_contradiction"] = component
        assignment.contradicted_attributes[feature] = MULTI_VALUE_SEPARATOR.join(outcome.candidate_values)
        reasons.append(f"{target_item.category} {feature} fallback contradiction")

    if assignment.contradicted_attributes and not saw_positive:
        assignment.status = "contradiction"
    assignment.score = round(score, 6)
    return _CandidateOption(
        assignment=assignment,
        score=score,
        score_breakdown=breakdown,
        reasons=reasons,
    )


def _build_missing_candidate(target_item: V3TargetItem) -> _CandidateOption:
    assignment = V3ItemAssignment(
        target_item_id=target_item.target_item_id,
        target_category=target_item.category,
        status="missing",
        source="none",
        score=MISSING_TARGET_PENALTY,
        missing_attributes=["category"],
        notes=["no matching extracted item or fallback category"],
    )
    return _CandidateOption(
        assignment=assignment,
        score=MISSING_TARGET_PENALTY,
        score_breakdown={f"item:{target_item.target_item_id}:missing_target": MISSING_TARGET_PENALTY},
        reasons=[f"{target_item.category} missing"],
    )


def _select_best_assignment(options_by_target: list[list[_CandidateOption]]) -> list[_CandidateOption]:
    best_choice: list[_CandidateOption] | None = None
    best_key: tuple[float, int, int] | None = None

    for combination in product(*options_by_target):
        used_item_ids = [option.used_item_id for option in combination if option.used_item_id]
        if len(set(used_item_ids)) != len(used_item_ids):
            continue

        score = sum(option.score for option in combination)
        matched_count = sum(
            option.assignment.status in {"exact", "partial", "fallback_match"}
            for option in combination
        )
        exact_count = sum(option.assignment.status == "exact" for option in combination)
        key = (round(score, 6), matched_count, exact_count)
        if best_key is None or key > best_key:
            best_key = key
            best_choice = list(combination)

    assert best_choice is not None
    return best_choice


def _priority_bonus(
    target_item: V3TargetItem,
    *,
    feature: str,
    exact: bool,
    breakdown: dict[str, float],
    scale: float = 1.0,
    prefix: str | None = None,
) -> float:
    base_prefix = prefix or f"item:{target_item.target_item_id}:{feature}"
    required = set(target_item.required_attributes)
    preferred = set(target_item.preferred_attributes)
    if feature in required:
        bonus = REQUIRED_EXACT_BONUS if exact else REQUIRED_PARTIAL_BONUS
        scaled = round(bonus * scale, 6)
        breakdown[f"{base_prefix}:required_bonus"] = scaled
        return scaled
    if feature in preferred:
        bonus = PREFERRED_EXACT_BONUS if exact else PREFERRED_PARTIAL_BONUS
        scaled = round(bonus * scale, 6)
        breakdown[f"{base_prefix}:preferred_bonus"] = scaled
        return scaled
    return 0.0


def _score_detail_consistency(
    parsed_query: V3ParsedQuery,
    document: V3ArchiveDocument,
) -> tuple[float, dict[str, float], list[str]]:
    score = 0.0
    breakdown: dict[str, float] = {}
    reasons: list[str] = []
    for target_item in parsed_query.target_items:
        signal = _evaluate_detail_color_signal(target_item, document.detail)
        if signal == "confirmed":
            score += DETAIL_CONFIRMATION_BONUS
            breakdown[f"detail:{target_item.target_item_id}:confirmation"] = DETAIL_CONFIRMATION_BONUS
            reasons.append(f"{target_item.category} detail confirmation")
        elif signal == "contradiction":
            score += DETAIL_CONTRADICTION_PENALTY
            breakdown[f"detail:{target_item.target_item_id}:contradiction"] = DETAIL_CONTRADICTION_PENALTY
            reasons.append(f"{target_item.category} detail contradiction")
    return score, breakdown, reasons


def _score_auxiliary_constraints(
    parsed_query: V3ParsedQuery,
    document: V3ArchiveDocument,
) -> tuple[float, dict[str, float], list[str]]:
    score = 0.0
    breakdown: dict[str, float] = {}
    reasons: list[str] = []

    for feature, query_values in parsed_query.global_constraints.items():
        outcome = _classify_values(query_values, _split_values(document.canonical_tags.get(feature, "")))
        if outcome.status == "exact":
            score += GLOBAL_CONSTRAINT_EXACT_BONUS
            breakdown[f"global_constraint:{feature}:exact"] = GLOBAL_CONSTRAINT_EXACT_BONUS
            reasons.append(f"global constraint {feature} exact")
        elif outcome.status == "partial":
            score += GLOBAL_CONSTRAINT_PARTIAL_BONUS
            breakdown[f"global_constraint:{feature}:partial"] = GLOBAL_CONSTRAINT_PARTIAL_BONUS
            reasons.append(f"global constraint {feature} partial")

    for feature, query_values in parsed_query.style_preferences.items():
        if feature in ITEM_LEVEL_QUERY_FEATURES or feature not in STYLE_PREFERENCE_ALLOWED_KEYS:
            continue
        candidate_values = _split_values(document.canonical_tags.get(feature, ""))
        if not candidate_values and feature == "style_tags":
            candidate_values = list(_aggregate_item_style_tags(document.items))
        outcome = _classify_values(query_values, candidate_values)
        if outcome.status == "exact":
            score += STYLE_PREFERENCE_EXACT_BONUS
            breakdown[f"style_preference:{feature}:exact"] = STYLE_PREFERENCE_EXACT_BONUS
            reasons.append(f"style preference {feature} exact")
        elif outcome.status == "partial":
            score += STYLE_PREFERENCE_PARTIAL_BONUS
            breakdown[f"style_preference:{feature}:partial"] = STYLE_PREFERENCE_PARTIAL_BONUS
            reasons.append(f"style preference {feature} partial")

    return score, breakdown, reasons


def _detect_cross_item_swap_targets(
    parsed_query: V3ParsedQuery,
    document: V3ArchiveDocument,
) -> set[str]:
    if len(parsed_query.target_items) < 2 or len(document.items) < 2:
        return set()

    flagged: set[str] = set()
    for target_item in parsed_query.target_items:
        same_category_items = [
            item for item in document.items if _normalize(item.category) == _normalize(target_item.category)
        ]
        other_items = [
            item for item in document.items if _normalize(item.category) != _normalize(target_item.category)
        ]
        if not same_category_items or not other_items:
            continue

        for feature in ATTRIBUTE_KEYS:
            query_values = list(getattr(target_item, feature))
            if not query_values:
                continue

            same_has_requested = any(
                _classify_values(query_values, list(getattr(item, feature))).status in {"exact", "partial"}
                for item in same_category_items
            )
            same_has_contradiction = any(
                _classify_values(query_values, list(getattr(item, feature))).status == "contradiction"
                for item in same_category_items
            )
            other_has_requested = any(
                _classify_values(query_values, list(getattr(item, feature))).status in {"exact", "partial"}
                for item in other_items
            )
            if not same_has_requested and same_has_contradiction and other_has_requested:
                flagged.add(target_item.target_item_id)
                break

    if len(flagged) >= 2:
        return flagged
    return set()


def _evaluate_detail_color_signal(target_item: V3TargetItem, detail: str) -> str:
    if not target_item.color or not detail.strip():
        return "none"

    normalized_query_colors = {_normalize(value) for value in target_item.color}
    normalized_category = _normalize(target_item.category)
    for phrase in _split_values(detail):
        normalized_phrase = _normalize(phrase)
        if normalized_category not in normalized_phrase:
            continue
        phrase_colors = _extract_pre_category_colors(normalized_phrase, normalized_category)
        if not phrase_colors:
            continue
        if phrase_colors & normalized_query_colors:
            if phrase_colors <= normalized_query_colors:
                return "confirmed"
            return "contradiction"
        return "contradiction"
    return "none"


def _extract_pre_category_colors(phrase: str, category: str) -> set[str]:
    prefix = phrase.split(category, 1)[0]
    words = set(prefix.split())
    return {color for color in COLOR_LEXICON if color in words}


def _aggregate_item_style_tags(items: Iterable[V3DocumentItem]) -> list[str]:
    aggregated: list[str] = []
    for item in items:
        for value in item.style_tags:
            normalized = value.strip()
            if normalized and normalized not in aggregated:
                aggregated.append(normalized)
    return aggregated


def _classify_values(query_values: list[str], candidate_values: list[str]) -> _MatchOutcome:
    normalized_query = [_normalize(value) for value in query_values if _normalize(value)]
    normalized_candidate = [_normalize(value) for value in candidate_values if _normalize(value)]
    if not normalized_candidate:
        return _MatchOutcome(
            status="missing",
            query_values=normalized_query,
            candidate_values=normalized_candidate,
        )

    overlap = [value for value in normalized_query if value in set(normalized_candidate)]
    if normalized_query == normalized_candidate:
        return _MatchOutcome(
            status="exact",
            overlap_values=overlap,
            query_values=normalized_query,
            candidate_values=normalized_candidate,
        )
    if overlap:
        return _MatchOutcome(
            status="partial",
            overlap_values=overlap,
            query_values=normalized_query,
            candidate_values=normalized_candidate,
        )
    return _MatchOutcome(
        status="contradiction",
        query_values=normalized_query,
        candidate_values=normalized_candidate,
    )


def _build_rank_explanation(
    item_assignments: Sequence[V3ItemAssignment],
    score_breakdown: dict[str, float],
) -> str:
    assignment_summary = ", ".join(
        (
            f"{assignment.target_item_id}:{assignment.target_category}->"
            f"{assignment.matched_category or 'none'}[{assignment.status}]"
        )
        for assignment in item_assignments
    ) or "none"
    score_summary = ", ".join(
        f"{label}={value:+.1f}" for label, value in score_breakdown.items()
    ) or "none"
    return f"item_assignments: {assignment_summary} | score_summary: {score_summary}"


def _split_values(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(MULTI_VALUE_SEPARATOR) if item.strip()]


def _normalize(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").split())


def _dedupe(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value and value not in deduped:
            deduped.append(value)
    return deduped
