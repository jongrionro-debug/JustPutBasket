from __future__ import annotations

import pytest

from switch_query.v3.models import V3ParsedQuery, V3TargetItem
from switch_query.v3.query_validation import ensure_valid_v3_parsed_query, validate_v3_parsed_query


def test_v3_query_validation_accepts_single_item_query() -> None:
    parsed_query = V3ParsedQuery(
        query_text="black relaxed trousers",
        target_items=[
            V3TargetItem(
                target_item_id="item_1",
                category="trousers",
                color=["black"],
                silhouette=["relaxed"],
                raw_phrase="black relaxed trousers",
            )
        ],
        confidence=0.9,
    )

    result = validate_v3_parsed_query(parsed_query)

    assert result.violations == []
    assert result.validated_query.target_items[0].category == "trousers"


def test_v3_query_validation_accepts_multi_item_query_with_bound_phrases() -> None:
    parsed_query = V3ParsedQuery(
        query_text="white trousers with vintage black jacket",
        target_items=[
            V3TargetItem(
                target_item_id="item_1",
                category="trousers",
                color=["white"],
                raw_phrase="white trousers",
            ),
            V3TargetItem(
                target_item_id="item_2",
                category="jacket",
                color=["black"],
                style_tags=["vintage"],
                raw_phrase="vintage black jacket",
            ),
        ],
        confidence=0.9,
    )

    result = validate_v3_parsed_query(parsed_query)

    assert result.violations == []
    assert len(result.validated_query.target_items) == 2


def test_v3_query_validation_requires_raw_phrase_for_multi_item_queries() -> None:
    parsed_query = V3ParsedQuery(
        query_text="white trousers with black jacket",
        target_items=[
            V3TargetItem(
                target_item_id="item_1",
                category="trousers",
                color=["white"],
                raw_phrase="",
            ),
            V3TargetItem(
                target_item_id="item_2",
                category="jacket",
                color=["black"],
                raw_phrase="black jacket",
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="ambiguous_attribute_binding"):
        ensure_valid_v3_parsed_query(parsed_query)


def test_v3_query_validation_rejects_cross_item_color_swap() -> None:
    parsed_query = V3ParsedQuery(
        query_text="white trousers with black jacket",
        target_items=[
            V3TargetItem(
                target_item_id="item_1",
                category="trousers",
                color=["white"],
                raw_phrase="black jacket",
            ),
            V3TargetItem(
                target_item_id="item_2",
                category="jacket",
                color=["black"],
                raw_phrase="white trousers",
            ),
        ],
    )

    with pytest.raises(RuntimeError, match="cross_item_attribute_swap"):
        ensure_valid_v3_parsed_query(parsed_query)


def test_v3_query_validation_rejects_orphan_top_level_attributes() -> None:
    parsed_query = V3ParsedQuery(
        query_text="black trousers",
        target_items=[
            V3TargetItem(
                target_item_id="item_1",
                category="trousers",
                color=["black"],
                raw_phrase="black trousers",
            )
        ],
    )

    with pytest.raises(RuntimeError, match="orphan_global_attribute"):
        ensure_valid_v3_parsed_query(parsed_query, payload={"target_items": [], "color": ["black"]})
