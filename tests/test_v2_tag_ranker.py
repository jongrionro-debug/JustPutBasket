from __future__ import annotations

from switch_query.v2 import (
    TagRanker,
    TagRankerConfig,
    V2ArchiveDocument,
    V2ParsedQuery,
    classify_feature_match,
)
from switch_query.v2.tag_ranker import _evaluate_detail_color_signal


def build_query(**canonical_tags: str) -> V2ParsedQuery:
    required_features = [feature for feature in ("category", "color") if feature in canonical_tags]
    preferred_features = [
        feature
        for feature in canonical_tags
        if feature not in required_features
    ]
    return V2ParsedQuery(
        query_text="test query",
        canonical_tags=canonical_tags,
        raw_phrases={feature: value for feature, value in canonical_tags.items()},
        required_features=required_features,
        preferred_features=preferred_features,
        confidence=0.9,
    )


def build_document(image_id: str, **canonical_tags: str) -> V2ArchiveDocument:
    return V2ArchiveDocument(
        image_id=image_id,
        file_path=f"/tmp/{image_id}.jpg",
        brand="brand",
        season_group="spring-ready-to-wear",
        canonical_tags=canonical_tags,
        raw_tags={feature: value for feature, value in canonical_tags.items()},
    )


def test_classify_feature_match_returns_partial_for_multi_value_overlap() -> None:
    match = classify_feature_match("category", "trousers", "shirt|trousers|shoes")

    assert match.status == "partial"
    assert match.overlap_values == ["trousers"]


def test_classify_feature_match_returns_contradiction_without_overlap() -> None:
    match = classify_feature_match("category", "trousers", "dress|heels")

    assert match.status == "contradiction"
    assert match.overlap_values == []


def test_classify_feature_match_returns_missing_for_empty_candidate_value() -> None:
    match = classify_feature_match("category", "trousers", "")

    assert match.status == "missing"


def test_tag_ranker_prefers_exact_match_over_partial_match() -> None:
    query = build_query(category="trousers", color="black", silhouette="relaxed", mood="minimal")
    documents = [
        build_document("exact", category="trousers", color="black", silhouette="relaxed", mood="minimal"),
        build_document(
            "partial",
            category="shirt|trousers|shoes",
            color="black",
            silhouette="relaxed|oversized",
            mood="minimal|modern",
        ),
    ]

    results = TagRanker(TagRankerConfig(top_k=2)).rank(query, documents)

    assert [result.image_id for result in results] == ["exact", "partial"]
    assert results[0].score > results[1].score
    assert results[0].score_breakdown["category:exact"] == 8.0
    assert results[1].score_breakdown["category:partial"] == 2.0
    assert results[1].score_breakdown["color:exact"] == 2.0
    assert results[1].score_breakdown["silhouette:partial"] == 2.0
    assert results[1].score_breakdown["mood:partial"] == 1.0


def test_tag_ranker_excludes_category_mismatch_with_hard_filter() -> None:
    query = build_query(category="trousers", color="black")
    documents = [
        build_document("good", category="shirt|trousers", color="black"),
        build_document("bad", category="dress|heels", color="black"),
        build_document("missing", color="black"),
    ]

    results = TagRanker(TagRankerConfig(top_k=5)).rank(query, documents)

    assert [result.image_id for result in results] == ["good"]


def test_tag_ranker_applies_required_missing_penalty_for_color() -> None:
    query = build_query(category="coat", color="black", mood="minimal")
    documents = [
        build_document("with-color", category="coat", color="black", mood="minimal"),
        build_document("missing-color", category="coat", mood="minimal"),
    ]

    results = TagRanker(TagRankerConfig(top_k=2)).rank(query, documents)

    assert [result.image_id for result in results] == ["with-color", "missing-color"]
    assert results[1].score_breakdown["color:missing"] == -10.0
    assert "color missing" in results[1].match_reasons


def test_tag_ranker_applies_color_contradiction_penalty() -> None:
    query = build_query(category="coat", color="black")
    documents = [
        build_document("exact-color", category="coat", color="black"),
        build_document("wrong-color", category="coat", color="white"),
    ]

    results = TagRanker(TagRankerConfig(top_k=2)).rank(query, documents)

    assert [result.image_id for result in results] == ["exact-color", "wrong-color"]
    assert results[1].score_breakdown["color:contradiction"] == -8.0
    assert results[1].mismatched_attributes["color"] == "white"


def test_tag_ranker_penalizes_detail_level_color_contradiction_for_target_category() -> None:
    query = build_query(category="trousers", color="black", silhouette="relaxed", mood="minimal")
    documents = [
        build_document(
            "good",
            category="tunic|trousers|shoes",
            color="black",
            silhouette="relaxed",
            mood="minimal",
            detail="long tunic|wide leg black trousers|beige pointed shoes",
        ),
        build_document(
            "bad",
            category="tunic|trousers|shoes",
            color="black",
            silhouette="relaxed",
            mood="minimal",
            detail="long tunic|wide leg white trousers|beige pointed shoes",
        ),
    ]

    results = TagRanker(TagRankerConfig(top_k=2)).rank(query, documents)

    assert [result.image_id for result in results] == ["good", "bad"]
    assert results[0].score == 15.0
    assert results[0].score_breakdown["color:exact"] == 4.0
    assert results[0].score_breakdown["color:detail_target_confirmation"] == 2.0
    assert results[1].score_breakdown["detail:target_color_contradiction"] == -8.0
    assert results[1].score_breakdown["color:exact"] == 2.0
    assert "wide leg white trousers" in results[1].mismatched_attributes["detail"]
    assert "detail target color contradiction" in results[1].match_reasons


def test_detail_color_signal_treats_mixed_color_trousers_as_contradiction() -> None:
    query = build_query(category="trousers", color="black")
    document = build_document(
        "mixed",
        category="jacket|trousers",
        color="black",
        detail="black leather jacket|black and white vertical striped trousers",
    )

    signal = _evaluate_detail_color_signal(query, document)

    assert signal.confirmed_phrase is None
    assert signal.contradiction_phrase is not None
    assert "black and white vertical striped trousers" in signal.contradiction_phrase


def test_detail_color_signal_ignores_post_category_accent_colors() -> None:
    query = build_query(category="trousers", color="black")
    document = build_document(
        "accent",
        category="top|trousers",
        color="black",
        detail="long top|wide leg denim trousers with black embroidery",
    )

    signal = _evaluate_detail_color_signal(query, document)

    assert signal.confirmed_phrase is None
    assert signal.contradiction_phrase is None
