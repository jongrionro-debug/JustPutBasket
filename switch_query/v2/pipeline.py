"""Runtime retrieval pipeline for the V2 text/tag retrieval system."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from .llm_parser import LuxiaQueryParser
from .models import (
    LocalIndexStore,
    QueryParser,
    TextEncoder,
    V2ArchiveIndex,
    V2PipelineInput,
    V2PipelineOutput,
)
from .tag_ranker import TagRanker, TagRankerConfig


@dataclass(slots=True)
class V2PipelineConfig:
    top_k: int = 20
    enable_rerank: bool = False


class V2Pipeline:
    def __init__(
        self,
        index_store: LocalIndexStore,
        *,
        encoder: TextEncoder | None = None,
        parser: QueryParser | None = None,
        ranker: TagRanker | None = None,
        config: V2PipelineConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.index_store = index_store
        self.parser = parser
        self.ranker = ranker
        self.config = config or V2PipelineConfig()

    def run(self, pipeline_input: V2PipelineInput) -> V2PipelineOutput:
        archive_index = self.index_store.load()
        if not archive_index.documents:
            raise ValueError("Archive index is empty. Build or load V2 archive documents first.")

        parser = self.parser or LuxiaQueryParser(archive_index.feature_vocabulary)
        parsed_query = parser.parse(
            pipeline_input.query_text,
            stage=pipeline_input.stage,
            balance_score=pipeline_input.balance_score,
            user_uploaded_image=pipeline_input.user_uploaded_image,
        )
        ranker = self.ranker or TagRanker(TagRankerConfig(top_k=self.config.top_k))
        results = ranker.rank(parsed_query, archive_index.documents)
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
                "ranking_mode": "tag_rank_first",
                "rerank_applied": False,
            },
        )


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
