"""Candidate retrieval for the V3 item-aware pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from .models import (
    MultimodalEncoder,
    V3CandidateMode,
    V3CandidateTrace,
    V3EmbeddingCandidate,
    V3IndexedDocument,
    V3ParsedQuery,
)
from .ranker import V3Ranker, V3RankerConfig


@dataclass(slots=True)
class V3RetrieverConfig:
    symbolic_candidate_k: int = 100
    embedding_candidate_k: int = 100


@dataclass(slots=True)
class V3RetrievedCandidates:
    documents: list[V3IndexedDocument]
    candidate_traces: list[V3CandidateTrace]
    symbolic_candidate_count: int
    embedding_candidate_count: int
    union_candidate_count: int


class V3Retriever:
    def __init__(
        self,
        *,
        encoder: MultimodalEncoder | None = None,
        ranker: V3Ranker | None = None,
        config: V3RetrieverConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.config = config or V3RetrieverConfig()
        self.ranker = ranker or V3Ranker(V3RankerConfig(top_k=self.config.symbolic_candidate_k))

    def retrieve(
        self,
        parsed_query: V3ParsedQuery,
        documents: Sequence[V3IndexedDocument],
        candidate_mode: V3CandidateMode,
    ) -> V3RetrievedCandidates:
        document_by_id = {document.image_id: document for document in documents}

        symbolic_results = []
        symbolic_traces: dict[str, V3CandidateTrace] = {}
        if candidate_mode in {"symbolic_only", "union"}:
            symbolic_ranker = V3Ranker(V3RankerConfig(top_k=self.config.symbolic_candidate_k))
            symbolic_results = symbolic_ranker.rank(parsed_query, documents)
            symbolic_traces = {
                result.image_id: V3CandidateTrace(
                    image_id=result.image_id,
                    in_symbolic_pool=True,
                    symbolic_rank=rank,
                    symbolic_score=result.score,
                )
                for rank, result in enumerate(symbolic_results, start=1)
            }

        embedding_candidates: list[V3EmbeddingCandidate] = []
        embedding_traces: dict[str, V3CandidateTrace] = {}
        if candidate_mode in {"embedding_only", "union"}:
            documents_with_vectors = [document for document in documents if document.vector]
            if not documents_with_vectors:
                raise ValueError(
                    "Embedding candidate retrieval requires at least one indexed document with vectors."
                )
            if self.encoder is None:
                raise ValueError("Embedding candidate retrieval requires a multimodal encoder.")

            serialized_query = serialize_parsed_query(parsed_query)
            # Precision-first baseline: encode only the parser-serialized query.
            # If we revisit dense recall later, add late fusion at the score level
            # rather than averaging raw and serialized query embeddings.
            query_vectors = self.encoder.encode_text([serialized_query])
            if len(query_vectors) != 1:
                raise RuntimeError("Text encoder output size did not match query count")
            query_vector = query_vectors[0]
            scored = [
                V3EmbeddingCandidate(
                    image_id=document.image_id,
                    embedding_score=round(dense_cosine_similarity(query_vector, document.vector), 6),
                    embedding_rank=0,
                )
                for document in documents_with_vectors
            ]
            scored.sort(key=lambda item: (-item.embedding_score, item.image_id))
            embedding_candidates = [
                V3EmbeddingCandidate(
                    image_id=item.image_id,
                    embedding_score=item.embedding_score,
                    embedding_rank=rank,
                )
                for rank, item in enumerate(scored[: self.config.embedding_candidate_k], start=1)
            ]
            embedding_traces = {
                item.image_id: V3CandidateTrace(
                    image_id=item.image_id,
                    in_embedding_pool=True,
                    embedding_rank=item.embedding_rank,
                    embedding_score=item.embedding_score,
                )
                for item in embedding_candidates
            }

        if candidate_mode == "symbolic_only":
            selected_ids = [result.image_id for result in symbolic_results]
            traces = list(symbolic_traces.values())
        elif candidate_mode == "embedding_only":
            selected_ids = [candidate.image_id for candidate in embedding_candidates]
            traces = list(embedding_traces.values())
        else:
            traces_by_id = dict(symbolic_traces)
            for image_id, trace in embedding_traces.items():
                existing = traces_by_id.get(image_id)
                if existing is None:
                    traces_by_id[image_id] = trace
                    continue
                existing.in_embedding_pool = True
                existing.embedding_rank = trace.embedding_rank
                existing.embedding_score = trace.embedding_score
            selected_ids = [result.image_id for result in symbolic_results]
            selected_ids.extend(
                candidate.image_id
                for candidate in embedding_candidates
                if candidate.image_id not in traces_by_id or not traces_by_id[candidate.image_id].in_symbolic_pool
            )
            deduped_ids: list[str] = []
            seen_ids: set[str] = set()
            for image_id in selected_ids:
                if image_id in seen_ids:
                    continue
                seen_ids.add(image_id)
                deduped_ids.append(image_id)
            selected_ids = deduped_ids
            traces = [traces_by_id[image_id] for image_id in selected_ids]

        selected_documents = [
            document_by_id[image_id]
            for image_id in selected_ids
            if image_id in document_by_id
        ]
        return V3RetrievedCandidates(
            documents=selected_documents,
            candidate_traces=traces,
            symbolic_candidate_count=len(symbolic_results),
            embedding_candidate_count=len(embedding_candidates),
            union_candidate_count=len(selected_documents),
        )


def serialize_parsed_query(parsed_query: V3ParsedQuery) -> str:
    parts: list[str] = []
    for index, item in enumerate(parsed_query.target_items, start=1):
        tokens = [f"item{index}", item.category]
        for feature in (
            "color",
            "silhouette",
            "material",
            "pattern",
            "texture",
            "style_concepts",
            "style_tags",
        ):
            values = list(getattr(item, feature))
            if values:
                tokens.append(f"{feature} {' '.join(values)}")
        parts.append(" ; ".join(tokens))

    if parsed_query.global_constraints:
        global_tokens = [
            f"{feature} {' '.join(values)}"
            for feature, values in parsed_query.global_constraints.items()
            if values
        ]
        if global_tokens:
            parts.append("global " + " ; ".join(global_tokens))

    if parsed_query.style_preferences:
        preference_tokens = [
            f"{feature} {' '.join(values)}"
            for feature, values in parsed_query.style_preferences.items()
            if values
        ]
        if preference_tokens:
            parts.append("preferences " + " ; ".join(preference_tokens))

    return " ; ".join(parts) if parts else parsed_query.query_text


def dense_cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) != len(right):
        raise ValueError("Dense vectors must have the same dimensionality")

    numerator = sum(lvalue * rvalue for lvalue, rvalue in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
