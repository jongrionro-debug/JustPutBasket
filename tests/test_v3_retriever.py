from __future__ import annotations

from switch_query.v3.models import V3IndexedDocument, V3ParsedQuery, V3TargetItem
from switch_query.v3.retriever import V3Retriever, V3RetrieverConfig


class FakeEncoder:
    def encode_text(self, texts):
        assert texts == ["item1 ; trousers ; color black"]
        return [[1.0, 0.0]]

    def encode_image(self, image_paths):
        raise AssertionError("encode_image should not be called during retrieval")


def build_query() -> V3ParsedQuery:
    return V3ParsedQuery(
        query_text="black trousers",
        target_items=[
            V3TargetItem(
                target_item_id="item_1",
                category="trousers",
                color=["black"],
                required_attributes=["category", "color"],
                raw_phrase="black trousers",
            )
        ],
        global_constraints={},
        style_preferences={},
        confidence=0.9,
    )


def build_document(
    image_id: str,
    *,
    detail: str,
    color: str,
    category: str = "trousers",
    vector: list[float] | None,
) -> V3IndexedDocument:
    return V3IndexedDocument(
        image_id=image_id,
        file_path=f"/tmp/{image_id}.jpg",
        brand="brand",
        season_group="spring-ready-to-wear",
        canonical_tags={"category": category, "color": color},
        raw_tags={},
        detail=detail,
        items=[],
        item_confidence=0.0,
        item_extraction_notes=[],
        vector=vector or [],
    )


def test_v3_retriever_symbolic_only_uses_existing_ranker_order() -> None:
    query = build_query()
    documents = [
        build_document("wrong", detail="white trousers", color="white", vector=[0.1, 1.0]),
        build_document("exact", detail="black trousers", color="black", vector=[1.0, 0.0]),
    ]

    result = V3Retriever(config=V3RetrieverConfig(symbolic_candidate_k=10)).retrieve(
        query,
        documents,
        "symbolic_only",
    )

    assert [document.image_id for document in result.documents] == ["exact", "wrong"]
    assert result.candidate_traces[0].image_id == "exact"
    assert result.candidate_traces[0].symbolic_rank == 1
    assert result.embedding_candidate_count == 0


def test_v3_retriever_embedding_only_orders_by_cosine_similarity() -> None:
    query = build_query()
    documents = [
        build_document("look-2", detail="white trousers", color="white", vector=[0.2, 1.0]),
        build_document("look-1", detail="black trousers", color="black", vector=[1.0, 0.0]),
    ]

    result = V3Retriever(
        encoder=FakeEncoder(),
        config=V3RetrieverConfig(embedding_candidate_k=10),
    ).retrieve(query, documents, "embedding_only")

    assert [document.image_id for document in result.documents] == ["look-1", "look-2"]
    assert result.candidate_traces[0].embedding_rank == 1
    assert result.candidate_traces[0].embedding_score == 1.0
    assert result.symbolic_candidate_count == 0


def test_v3_retriever_union_dedupes_and_keeps_pool_provenance() -> None:
    query = build_query()
    documents = [
        build_document("exact", detail="black trousers", color="black", vector=[1.0, 0.0]),
        build_document("extra", detail="unrelated", color="white", category="coat", vector=[0.8, 0.0]),
    ]

    result = V3Retriever(
        encoder=FakeEncoder(),
        config=V3RetrieverConfig(symbolic_candidate_k=1, embedding_candidate_k=10),
    ).retrieve(query, documents, "union")

    assert [document.image_id for document in result.documents] == ["exact", "extra"]
    assert result.candidate_traces[0].in_symbolic_pool is True
    assert result.candidate_traces[0].in_embedding_pool is True
    assert result.candidate_traces[1].in_symbolic_pool is False
    assert result.candidate_traces[1].in_embedding_pool is True


def test_v3_retriever_requires_vectors_for_embedding_modes() -> None:
    query = build_query()
    documents = [
        build_document("look-1", detail="black trousers", color="black", vector=None),
    ]

    try:
        V3Retriever(encoder=FakeEncoder()).retrieve(query, documents, "embedding_only")
    except ValueError as exc:
        assert "requires at least one indexed document with vectors" in str(exc)
    else:
        raise AssertionError("Expected embedding retrieval without vectors to fail")
