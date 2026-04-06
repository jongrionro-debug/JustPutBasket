from __future__ import annotations

from switch_query.v3 import (
    V3ArchiveDocument,
    V3DocumentItem,
    V3ParsedQuery,
    V3RankedResult,
    V3Ranker,
    V3RankerConfig,
    V3TargetItem,
)


def build_query(
    *,
    query_text: str,
    target_items: list[V3TargetItem],
    global_constraints: dict[str, list[str]] | None = None,
    style_preferences: dict[str, list[str]] | None = None,
) -> V3ParsedQuery:
    return V3ParsedQuery(
        query_text=query_text,
        target_items=target_items,
        global_constraints=global_constraints or {},
        style_preferences=style_preferences or {},
        confidence=0.9,
    )


def build_target_item(
    item_id: str,
    *,
    category: str,
    color: list[str] | None = None,
    silhouette: list[str] | None = None,
    style_tags: list[str] | None = None,
    required_attributes: list[str] | None = None,
    preferred_attributes: list[str] | None = None,
    raw_phrase: str | None = None,
) -> V3TargetItem:
    return V3TargetItem(
        target_item_id=item_id,
        category=category,
        color=color or [],
        silhouette=silhouette or [],
        style_tags=style_tags or [],
        required_attributes=required_attributes or [],
        preferred_attributes=preferred_attributes or [],
        raw_phrase=raw_phrase or category,
    )


def build_item(
    item_id: str,
    *,
    category: str,
    color: list[str] | None = None,
    silhouette: list[str] | None = None,
    style_tags: list[str] | None = None,
) -> V3DocumentItem:
    return V3DocumentItem(
        item_id=item_id,
        category=category,
        color=color or [],
        silhouette=silhouette or [],
        style_tags=style_tags or [],
        confidence=0.9,
        evidence=[],
        source="test_fixture",
    )


def build_document(
    image_id: str,
    *,
    canonical_tags: dict[str, str] | None = None,
    detail: str = "",
    items: list[V3DocumentItem] | None = None,
) -> V3ArchiveDocument:
    return V3ArchiveDocument(
        image_id=image_id,
        file_path=f"/tmp/{image_id}.jpg",
        brand="brand",
        season_group="spring-ready-to-wear",
        canonical_tags=canonical_tags or {},
        raw_tags={},
        detail=detail,
        items=items or [],
        item_confidence=0.9,
        item_extraction_notes=[],
    )


def rank_documents(query: V3ParsedQuery, documents: list[V3ArchiveDocument]) -> list[V3RankedResult]:
    return V3Ranker(V3RankerConfig(top_k=10)).rank(query, documents)


def test_v3_ranker_prefers_exact_item_match_over_outfit_level_similarity() -> None:
    query = build_query(
        query_text="black relaxed trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                silhouette=["relaxed"],
                required_attributes=["category", "color", "silhouette"],
                raw_phrase="black relaxed trousers",
            )
        ],
    )
    exact_document = build_document(
        "exact",
        canonical_tags={"category": "trousers", "color": "black"},
        detail="relaxed black trousers",
        items=[
            build_item(
                "exact#1",
                category="trousers",
                color=["black"],
                silhouette=["relaxed"],
            )
        ],
    )
    wrong_document = build_document(
        "wrong",
        canonical_tags={"category": "top|trousers", "color": "black"},
        detail="black top|relaxed white trousers",
        items=[
            build_item("wrong#1", category="top", color=["black"]),
            build_item("wrong#2", category="trousers", color=["white"], silhouette=["relaxed"]),
        ],
    )

    results = rank_documents(query, [wrong_document, exact_document])

    assert [result.image_id for result in results] == ["exact", "wrong"]
    assert results[0].item_assignments[0].status == "exact"
    assert results[1].item_assignments[0].source == "item"
    assert results[1].item_assignments[0].status == "contradiction"
    assert results[1].item_assignments[0].contradicted_attributes["color"] == "white"
    assert not any("fallback" in key for key in results[1].score_breakdown)


def test_v3_ranker_prefers_full_multi_item_match_over_partial_result() -> None:
    query = build_query(
        query_text="white trousers with black jacket",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["white"],
                required_attributes=["category", "color"],
                raw_phrase="white trousers",
            ),
            build_target_item(
                "item_2",
                category="jacket",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black jacket",
            ),
        ],
    )
    full_document = build_document(
        "full",
        items=[
            build_item("full#1", category="trousers", color=["white"]),
            build_item("full#2", category="jacket", color=["black"]),
        ],
    )
    partial_document = build_document(
        "partial",
        items=[build_item("partial#1", category="trousers", color=["white"])],
    )

    results = rank_documents(query, [partial_document, full_document])

    assert [result.image_id for result in results] == ["full", "partial"]
    assert results[0].score_breakdown["coverage:full_item_set"] == 12.0
    assert [assignment.status for assignment in results[1].item_assignments] == ["exact", "missing"]


def test_v3_ranker_marks_cross_item_swap_as_hard_fail() -> None:
    query = build_query(
        query_text="white trousers with black jacket",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["white"],
                required_attributes=["category", "color"],
                raw_phrase="white trousers",
            ),
            build_target_item(
                "item_2",
                category="jacket",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black jacket",
            ),
        ],
    )
    good_document = build_document(
        "good",
        items=[
            build_item("good#1", category="trousers", color=["white"]),
            build_item("good#2", category="jacket", color=["black"]),
        ],
    )
    swap_document = build_document(
        "swap",
        items=[
            build_item("swap#1", category="trousers", color=["black"]),
            build_item("swap#2", category="jacket", color=["white"]),
        ],
    )

    results = rank_documents(query, [swap_document, good_document])

    assert [result.image_id for result in results] == ["good", "swap"]
    assert results[1].score_breakdown["penalty:cross_item_swap"] == -100.0
    assert all(assignment.status == "hard_fail" for assignment in results[1].item_assignments)
    assert "cross-item attribute swap detected" in results[1].match_reasons


def test_v3_ranker_uses_fallback_only_when_items_are_missing() -> None:
    query = build_query(
        query_text="black trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black trousers",
            )
        ],
    )
    fallback_document = build_document(
        "fallback",
        canonical_tags={"category": "top|trousers", "color": "black"},
        detail="relaxed black trousers",
    )
    missing_document = build_document(
        "missing",
        canonical_tags={"category": "coat", "color": "black"},
        detail="black coat",
    )

    results = rank_documents(query, [missing_document, fallback_document])

    assert [result.image_id for result in results] == ["fallback", "missing"]
    assert results[0].item_assignments[0].status == "fallback_match"
    assert results[0].item_assignments[0].source == "fallback"


def test_v3_ranker_does_not_override_extracted_item_with_fallback_tags() -> None:
    query = build_query(
        query_text="black trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black trousers",
            )
        ],
    )
    contradictory_document = build_document(
        "contradictory",
        canonical_tags={"category": "trousers", "color": "black"},
        detail="black look with white trousers",
        items=[build_item("contradictory#1", category="trousers", color=["white"])],
    )

    result = rank_documents(query, [contradictory_document])[0]

    assert result.item_assignments[0].source == "item"
    assert result.item_assignments[0].status == "contradiction"
    assert not any("fallback" in key for key in result.score_breakdown)


def test_v3_ranker_respects_required_attributes_more_than_non_required() -> None:
    query = build_query(
        query_text="black trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black trousers",
            )
        ],
    )
    matched_document = build_document(
        "matched",
        items=[build_item("matched#1", category="trousers", color=["black"])],
    )
    missing_color_document = build_document(
        "missing-color",
        items=[build_item("missing-color#1", category="trousers")],
    )

    results = rank_documents(query, [missing_color_document, matched_document])

    assert [result.image_id for result in results] == ["matched", "missing-color"]
    assert results[0].score_breakdown["item:item_1:color:required_bonus"] == 2.0
    assert results[1].score_breakdown["item:item_1:color:missing_required"] == -6.0


def test_v3_ranker_uses_preferred_attributes_as_soft_bonus() -> None:
    query = build_query(
        query_text="black trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                style_tags=["minimal"],
                required_attributes=["category", "color"],
                preferred_attributes=["style_tags"],
                raw_phrase="black minimal trousers",
            )
        ],
    )
    preferred_document = build_document(
        "preferred",
        items=[
            build_item(
                "preferred#1",
                category="trousers",
                color=["black"],
                style_tags=["minimal"],
            )
        ],
    )
    plain_document = build_document(
        "plain",
        items=[build_item("plain#1", category="trousers", color=["black"])],
    )

    results = rank_documents(query, [plain_document, preferred_document])

    assert [result.image_id for result in results] == ["preferred", "plain"]
    assert results[0].score_breakdown["item:item_1:style_tags:preferred_bonus"] == 1.0
    assert results[1].item_assignments[0].status == "partial"


def test_v3_ranker_applies_global_and_style_bonus_as_auxiliary_signal() -> None:
    query = build_query(
        query_text="black trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black trousers",
            )
        ],
        global_constraints={"mood": ["minimal"]},
        style_preferences={"era": ["modern"]},
    )
    enriched_document = build_document(
        "enriched",
        canonical_tags={"mood": "minimal", "era": "modern"},
        items=[build_item("enriched#1", category="trousers", color=["black"])],
    )
    plain_document = build_document(
        "plain",
        items=[build_item("plain#1", category="trousers", color=["black"])],
    )

    results = rank_documents(query, [plain_document, enriched_document])

    assert [result.image_id for result in results] == ["enriched", "plain"]
    assert results[0].score_breakdown["global_constraint:mood:exact"] == 1.5
    assert results[0].score_breakdown["style_preference:era:exact"] == 1.0


def test_v3_ranker_ignores_item_level_features_inside_style_preferences() -> None:
    query = build_query(
        query_text="black relaxed trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                raw_phrase="black relaxed trousers",
            )
        ],
        style_preferences={
            "color": ["black"],
            "silhouette": ["relaxed"],
            "category": ["trousers"],
            "era": ["modern"],
        },
    )
    document = build_document(
        "styled",
        canonical_tags={
            "category": "trousers",
            "color": "black",
            "silhouette": "relaxed",
            "era": "modern",
        },
        items=[build_item("styled#1", category="trousers", color=["black"], silhouette=["relaxed"])],
    )

    result = rank_documents(query, [document])[0]

    assert result.score_breakdown["style_preference:era:exact"] == 1.0
    assert "style_preference:color:exact" not in result.score_breakdown
    assert "style_preference:silhouette:exact" not in result.score_breakdown
    assert "style_preference:category:partial" not in result.score_breakdown


def test_v3_ranker_uses_detail_as_auxiliary_confirmation_and_penalty() -> None:
    query = build_query(
        query_text="black trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black trousers",
            )
        ],
    )
    confirmed_document = build_document(
        "confirmed",
        canonical_tags={"category": "trousers", "color": "black"},
        detail="relaxed black trousers",
    )
    contradicted_document = build_document(
        "contradicted",
        canonical_tags={"category": "trousers", "color": "black"},
        detail="relaxed white trousers",
    )

    results = rank_documents(query, [contradicted_document, confirmed_document])

    assert [result.image_id for result in results] == ["confirmed", "contradicted"]
    assert results[0].score_breakdown["detail:item_1:confirmation"] == 1.5
    assert results[1].score_breakdown["detail:item_1:contradiction"] == -1.5


def test_v3_ranker_uses_v2_style_tie_breaking_for_identical_scores() -> None:
    query = build_query(
        query_text="black trousers",
        target_items=[
            build_target_item(
                "item_1",
                category="trousers",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black trousers",
            )
        ],
    )
    document_a = build_document(
        "a-look",
        items=[build_item("a-look#1", category="trousers", color=["black"])],
    )
    document_b = build_document(
        "b-look",
        items=[build_item("b-look#1", category="trousers", color=["black"])],
    )

    results = rank_documents(query, [document_b, document_a])

    assert [result.image_id for result in results] == ["a-look", "b-look"]
    assert results[0].score == results[1].score
