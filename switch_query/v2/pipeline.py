"""Runtime retrieval pipeline for the V2 text/tag retrieval system."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .explanation import explain_match
from .models import (
    LocalIndexStore,
    TextEncoder,
    V2ArchiveIndex,
    V2PipelineInput,
    V2PipelineOutput,
    V2RankedResult,
)
from .parser import V2QueryParser


@dataclass(slots=True)
class V2PipelineConfig:
    top_k: int = 20


class V2Pipeline:
    def __init__(
        self,
        encoder: TextEncoder,
        index_store: LocalIndexStore,
        *,
        parser: V2QueryParser | None = None,
        config: V2PipelineConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.index_store = index_store
        self.parser = parser
        self.config = config or V2PipelineConfig()

    def run(self, pipeline_input: V2PipelineInput) -> V2PipelineOutput:
        archive_index = self.index_store.load()
        if not archive_index.documents:
            raise ValueError("Archive index is empty. Build or load V2 archive documents first.")

        parser = self.parser or V2QueryParser(archive_index.feature_vocabulary)
        parsed_query = parser.parse(
            pipeline_input.query_text,
            stage=pipeline_input.stage,
            balance_score=pipeline_input.balance_score,
            user_uploaded_image=pipeline_input.user_uploaded_image,
        )
        query_vector = self.encoder.encode_text([parsed_query.query_document])[0]
        results = self._rank_documents(archive_index, parsed_query.canonical_tags, query_vector)
        return V2PipelineOutput(
            parsed_query=parsed_query,
            top_results=results,
            retrieval_metadata={
                "top_k": self.config.top_k,
                "indexed_document_count": len(archive_index.documents),
                "stage": pipeline_input.stage,
                "balance_score": pipeline_input.balance_score,
                "used_uploaded_image": bool(pipeline_input.user_uploaded_image),
                "uploaded_image_used_in_scoring": False,
            },
        )

    def _rank_documents(
        self,
        archive_index: V2ArchiveIndex,
        query_tags: dict[str, str],
        query_vector: list[float],
    ) -> list[V2RankedResult]:
        scored: list[tuple[float, V2RankedResult]] = []
        for document in archive_index.documents:
            score = dense_cosine_similarity(query_vector, document.vector)
            matched, mismatched, missing, explanation = explain_match(
                query_tags,
                document.canonical_tags,
            )
            scored.append(
                (
                    score,
                    V2RankedResult(
                        image_id=document.image_id,
                        file_path=document.file_path,
                        brand=document.brand,
                        season_group=document.season_group,
                        score=round(score, 6),
                        matched_attributes=matched,
                        mismatched_attributes=mismatched,
                        missing_attributes=missing,
                        explanation=explanation,
                    ),
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1].image_id))
        return [item[1] for item in scored[: self.config.top_k]]


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
