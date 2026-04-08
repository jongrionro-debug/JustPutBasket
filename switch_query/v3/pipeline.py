"""Runtime retrieval pipeline for the V3 item-aware retrieval system."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .llm_parser import LuxiaV3QueryParser
from .models import (
    LocalIndexStore,
    MultimodalEncoder,
    V3CandidateMode,
    V3CandidateTrace,
    V3PipelineInput,
    V3PipelineOutput,
)
from .ranker import V3Ranker, V3RankerConfig
from .retriever import V3Retriever, V3RetrieverConfig, serialize_parsed_query


@dataclass(slots=True)
class V3PipelineConfig:
    top_k: int = 20
    candidate_mode: V3CandidateMode = "union"
    symbolic_candidate_k: int = 100
    embedding_candidate_k: int = 100


class V3Pipeline:
    def __init__(
        self,
        index_store: LocalIndexStore,
        *,
        encoder: MultimodalEncoder | None = None,
        parser=None,
        retriever: V3Retriever | None = None,
        ranker: V3Ranker | None = None,
        config: V3PipelineConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.index_store = index_store
        self.parser = parser
        self.retriever = retriever
        self.ranker = ranker
        self.config = config or V3PipelineConfig()

    def run(self, pipeline_input: V3PipelineInput) -> V3PipelineOutput:
        archive_index = self.index_store.load()
        if not archive_index.documents:
            raise ValueError("Archive index is empty. Build or load V3 archive documents first.")

        parser = self.parser or LuxiaV3QueryParser()
        parsed_query = parser.parse(
            pipeline_input.query_text,
            stage=pipeline_input.stage,
            balance_score=pipeline_input.balance_score,
            user_uploaded_image=pipeline_input.user_uploaded_image,
        )
        retriever = self.retriever or V3Retriever(
            encoder=self.encoder,
            config=V3RetrieverConfig(
                symbolic_candidate_k=self.config.symbolic_candidate_k,
                embedding_candidate_k=self.config.embedding_candidate_k,
            ),
        )
        candidate_set = retriever.retrieve(
            parsed_query,
            archive_index.documents,
            self.config.candidate_mode,
        )
        ranker = self.ranker or V3Ranker(V3RankerConfig(top_k=self.config.top_k))
        ranked_results = ranker.rank(parsed_query, candidate_set.documents)
        top_results = _apply_hybrid_weighting(ranked_results, candidate_set.candidate_traces)[: self.config.top_k]
        candidate_traces = _apply_final_ranks(candidate_set.candidate_traces, top_results)

        retrieval_metadata = {
            "candidate_mode": self.config.candidate_mode,
            "top_k": self.config.top_k,
            "indexed_document_count": len(archive_index.documents),
            "documents_with_vectors": sum(1 for document in archive_index.documents if document.vector),
            "symbolic_candidate_count": candidate_set.symbolic_candidate_count,
            "embedding_candidate_count": candidate_set.embedding_candidate_count,
            "union_candidate_count": candidate_set.union_candidate_count,
            "final_ranking_mode": "hybrid_weighted",
            "serialized_query": serialize_parsed_query(parsed_query),
            "embedding_query_strategy": "serialized_only",
            "embedding_query_future_work": "late_fusion_planned",
            "uploaded_image_used_in_scoring": False,
        }
        return V3PipelineOutput(
            parsed_query=parsed_query,
            top_results=top_results,
            candidate_traces=candidate_traces,
            retrieval_metadata=retrieval_metadata,
        )


def _apply_final_ranks(
    candidate_traces: list[V3CandidateTrace],
    top_results,
) -> list[V3CandidateTrace]:
    final_by_image_id = {
        result.image_id: (rank, result.score)
        for rank, result in enumerate(top_results, start=1)
    }
    updated_traces: list[V3CandidateTrace] = []
    for trace in candidate_traces:
        final_rank, final_score = final_by_image_id.get(trace.image_id, (None, None))
        updated_traces.append(
            V3CandidateTrace(
                image_id=trace.image_id,
                in_symbolic_pool=trace.in_symbolic_pool,
                in_embedding_pool=trace.in_embedding_pool,
                symbolic_rank=trace.symbolic_rank,
                symbolic_score=trace.symbolic_score,
                embedding_rank=trace.embedding_rank,
                embedding_score=trace.embedding_score,
                final_rank=final_rank,
                final_score=final_score,
            )
        )
    updated_traces.sort(
        key=lambda trace: (
            trace.final_rank is None,
            trace.final_rank if trace.final_rank is not None else 10**9,
            trace.symbolic_rank if trace.symbolic_rank is not None else 10**9,
            trace.embedding_rank if trace.embedding_rank is not None else 10**9,
            trace.image_id,
        )
    )
    return updated_traces


def _apply_hybrid_weighting(
    ranked_results,
    candidate_traces: list[V3CandidateTrace],
):
    if not ranked_results:
        return []

    embedding_by_image_id = {
        trace.image_id: float(trace.embedding_score or 0.0)
        for trace in candidate_traces
    }
    symbolic_scores = [result.score for result in ranked_results]
    embedding_scores = [embedding_by_image_id.get(result.image_id, 0.0) for result in ranked_results]
    concept_scores = [_concept_support_score(result) for result in ranked_results]
    detail_scores = [_detail_support_score(result) for result in ranked_results]

    normalized_symbolic = _min_max_normalize(symbolic_scores)
    normalized_embedding = _min_max_normalize(embedding_scores)
    normalized_concept = _min_max_normalize(concept_scores)
    normalized_detail = _min_max_normalize(detail_scores)

    hybrid_results = []
    for index, result in enumerate(ranked_results):
        symbolic_component = normalized_symbolic[index]
        embedding_component = normalized_embedding[index]
        concept_component = normalized_concept[index]
        detail_component = normalized_detail[index]
        final_score = round(
            0.60 * symbolic_component
            + 0.25 * embedding_component
            + 0.10 * concept_component
            + 0.05 * detail_component,
            6,
        )
        updated_breakdown = dict(result.score_breakdown)
        updated_breakdown["hybrid:symbolic_raw_score"] = round(result.score, 6)
        updated_breakdown["hybrid:normalized_symbolic_item_score"] = round(symbolic_component, 6)
        updated_breakdown["hybrid:normalized_embedding_score"] = round(embedding_component, 6)
        updated_breakdown["hybrid:concept_support_score"] = round(concept_component, 6)
        updated_breakdown["hybrid:detail_consistency_score"] = round(detail_component, 6)
        updated_breakdown["hybrid:final_weighted_score"] = final_score
        hybrid_results.append(
            replace(
                result,
                score=final_score,
                score_breakdown=updated_breakdown,
                explanation=(
                    f"{result.explanation} | hybrid(symbolic={symbolic_component:.3f}, "
                    f"embedding={embedding_component:.3f}, concept={concept_component:.3f}, "
                    f"detail={detail_component:.3f})"
                ),
            )
        )

    hybrid_results.sort(key=lambda result: (-result.score, result.image_id))
    return hybrid_results


def _concept_support_score(result) -> float:
    return sum(
        value
        for key, value in result.score_breakdown.items()
        if ":style_concepts:" in key
    )


def _detail_support_score(result) -> float:
    return sum(
        value
        for key, value in result.score_breakdown.items()
        if key.startswith("detail:")
    )


def _min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        return [1.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]
