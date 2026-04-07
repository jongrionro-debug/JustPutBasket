from __future__ import annotations

from dataclasses import dataclass

from switch_query.v3.models import (
    V3ArchiveIndex,
    V3CandidateTrace,
    V3IndexedDocument,
    V3ParsedQuery,
    V3PipelineInput,
    V3RankedResult,
    V3TargetItem,
)
from switch_query.v3.pipeline import V3Pipeline, V3PipelineConfig
from switch_query.v3.retriever import V3RetrievedCandidates


class FakeIndexStore:
    def __init__(self, index: V3ArchiveIndex) -> None:
        self.index = index

    def load(self) -> V3ArchiveIndex:
        return self.index


class FakeParser:
    def __init__(self) -> None:
        self.calls = []

    def parse(self, query_text, *, stage, balance_score, user_uploaded_image=None):
        self.calls.append((query_text, stage, balance_score, user_uploaded_image))
        return V3ParsedQuery(
            query_text=query_text,
            target_items=[
                V3TargetItem(
                    target_item_id="item_1",
                    category="trousers",
                    color=["black"],
                    raw_phrase="black trousers",
                )
            ],
            global_constraints={},
            style_preferences={},
            confidence=0.9,
        )


class FakeRetriever:
    def __init__(self, documents: list[V3IndexedDocument]) -> None:
        self.documents = documents
        self.calls = []

    def retrieve(self, parsed_query, documents, candidate_mode):
        self.calls.append((parsed_query.query_text, len(documents), candidate_mode))
        return V3RetrievedCandidates(
            documents=self.documents,
            candidate_traces=[
                V3CandidateTrace(
                    image_id=self.documents[0].image_id,
                    in_symbolic_pool=True,
                    symbolic_rank=1,
                    symbolic_score=10.0,
                )
            ],
            symbolic_candidate_count=1,
            embedding_candidate_count=0,
            union_candidate_count=1,
        )


class FakeRanker:
    def __init__(self) -> None:
        self.calls = []

    def rank(self, parsed_query, documents):
        self.calls.append((parsed_query.query_text, [document.image_id for document in documents]))
        return [
            V3RankedResult(
                image_id=documents[0].image_id,
                file_path=documents[0].file_path,
                brand=documents[0].brand,
                season_group=documents[0].season_group,
                score=42.0,
                explanation="ranked by fake ranker",
            )
        ]


def build_document(image_id: str, vector: list[float] | None = None) -> V3IndexedDocument:
    return V3IndexedDocument(
        image_id=image_id,
        file_path=f"/tmp/{image_id}.jpg",
        brand="brand",
        season_group="spring-ready-to-wear",
        canonical_tags={"category": "trousers", "color": "black"},
        raw_tags={},
        detail="black trousers",
        items=[],
        item_confidence=0.0,
        item_extraction_notes=[],
        vector=vector or [],
    )


def test_v3_pipeline_uses_candidate_pool_and_preserves_metadata() -> None:
    indexed_documents = [build_document("look-1", vector=[1.0, 0.0])]
    parser = FakeParser()
    retriever = FakeRetriever(indexed_documents)
    ranker = FakeRanker()
    pipeline = V3Pipeline(
        index_store=FakeIndexStore(V3ArchiveIndex(documents=indexed_documents)),
        parser=parser,
        retriever=retriever,
        ranker=ranker,
        config=V3PipelineConfig(top_k=20, candidate_mode="union"),
    )

    output = pipeline.run(
        V3PipelineInput(
            query_text="black trousers",
            stage="mood_board",
            balance_score=0.0,
            user_uploaded_image="/tmp/uploaded.jpg",
        )
    )

    assert parser.calls == [("black trousers", "mood_board", 0.0, "/tmp/uploaded.jpg")]
    assert retriever.calls == [("black trousers", 1, "union")]
    assert ranker.calls == [("black trousers", ["look-1"])]
    assert output.top_results[0].score == 1.0
    assert output.candidate_traces[0].final_rank == 1
    assert output.candidate_traces[0].final_score == 1.0
    assert output.retrieval_metadata["candidate_mode"] == "union"
    assert output.retrieval_metadata["documents_with_vectors"] == 1
    assert output.retrieval_metadata["final_ranking_mode"] == "hybrid_weighted"
    assert output.retrieval_metadata["uploaded_image_used_in_scoring"] is False
