"""V1 image retrieval scaffold aligned with the planning document."""

from __future__ import annotations

from math import sqrt
from typing import Sequence

from switch_query.tagging.preprocessing import InventoryRow

from .models import (
    GeneratedReference,
    LocalIndexStore,
    MultimodalEncoder,
    SyntheticImageGenerator,
    V1ArchiveRecord,
    V1PipelineInput,
    V1PipelineOutput,
    V1QueryArtifacts,
    V1RankedResult,
    V1ScoredCandidate,
)
from .policies import (
    V1PipelineConfig,
    average_dense_vectors,
    balance_bucket,
    synthetic_reference_count,
)


class V1Pipeline:
    def __init__(
        self,
        encoder: MultimodalEncoder,
        image_generator: SyntheticImageGenerator,
        index_store: LocalIndexStore,
        config: V1PipelineConfig | None = None,
    ) -> None:
        self.encoder = encoder
        self.image_generator = image_generator
        self.index_store = index_store
        self.config = config or V1PipelineConfig()

    def build_archive_index(self, inventory_rows: Sequence[InventoryRow]) -> list[V1ArchiveRecord]:
        image_paths = [row.file_path for row in inventory_rows]
        vectors = self.encoder.encode_image(image_paths)
        if len(vectors) != len(inventory_rows):
            raise RuntimeError("Image encoder output size did not match archive inventory")

        records = [
            V1ArchiveRecord(
                image_id=row.image_id,
                file_path=row.file_path,
                brand=row.brand,
                vector=vector,
                metadata={
                    "year": row.year,
                    "season_group": row.season_group,
                    "source_type": row.source_type,
                    "filename": row.filename,
                },
            )
            for row, vector in zip(inventory_rows, vectors, strict=True)
        ]
        self.index_store.save(records)
        return records

    def run(self, pipeline_input: V1PipelineInput) -> V1PipelineOutput:
        archive_records = self.index_store.load()
        if not archive_records:
            raise ValueError("Archive index is empty. Build or load archive vectors before running.")

        generated_references = self._resolve_references(pipeline_input)
        text_vector = self.encoder.encode_text([pipeline_input.query_text])[0]
        generated_vectors = (
            self.encoder.encode_image([ref.image_path for ref in generated_references])
            if generated_references
            else []
        )
        merged_generated_vector = average_dense_vectors(generated_vectors)
        query_artifacts = V1QueryArtifacts(
            query_text=pipeline_input.query_text,
            text_vector=text_vector,
            generated_image_vectors=generated_vectors,
            merged_generated_image_vector=merged_generated_vector,
        )
        ranked = self._rank_archive(
            pipeline_input=pipeline_input,
            archive_records=archive_records,
            query_artifacts=query_artifacts,
        )
        return V1PipelineOutput(
            retrieval_mode=self.config.retrieval_mode,
            generated_references=generated_references,
            query_artifacts=query_artifacts,
            archive_results=ranked,
        )

    def _resolve_references(self, pipeline_input: V1PipelineInput) -> list[GeneratedReference]:
        if pipeline_input.generated_image_paths:
            return self._build_references_from_paths(
                pipeline_input.query_text,
                pipeline_input.balance_score,
                pipeline_input.generated_image_paths,
            )
        if self.config.retrieval_mode == "text_only":
            return []
        return self._generate_references(pipeline_input)

    def _generate_references(self, pipeline_input: V1PipelineInput) -> list[GeneratedReference]:
        count = synthetic_reference_count(pipeline_input.balance_score)
        bucket = balance_bucket(pipeline_input.balance_score)
        generated_paths = self.image_generator.generate(pipeline_input.query_text, count)
        if len(generated_paths) != count:
            raise RuntimeError("Synthetic image generator output size did not match requested count")
        return self._build_references_from_paths(
            pipeline_input.query_text,
            pipeline_input.balance_score,
            generated_paths,
        )

    def _build_references_from_paths(
        self,
        query_text: str,
        balance_score: float,
        generated_paths: Sequence[str],
    ) -> list[GeneratedReference]:
        bucket = balance_bucket(balance_score)
        return [
            GeneratedReference(
                generated_id=f"gen-{bucket}-{index + 1}",
                prompt_summary=query_text,
                image_path=image_path,
                balance_bucket=bucket,
            )
            for index, image_path in enumerate(generated_paths)
        ]

    def _rank_archive(
        self,
        pipeline_input: V1PipelineInput,
        archive_records: list[V1ArchiveRecord],
        query_artifacts: V1QueryArtifacts,
    ) -> list[V1RankedResult]:
        scored: list[V1ScoredCandidate] = []
        for record in archive_records:
            text_score = dense_cosine_similarity(query_artifacts.text_vector, record.vector)
            image_score = dense_cosine_similarity(
                query_artifacts.merged_generated_image_vector,
                record.vector,
            )
            final_score = self._compose_final_score(text_score, image_score)
            scored.append(
                V1ScoredCandidate(
                    image_id=record.image_id,
                    file_path=record.file_path,
                    brand=record.brand,
                    final_score=round(final_score, 6),
                    text_score=round(text_score, 6),
                    image_score=round(image_score, 6),
                    metadata=record.metadata,
                )
            )

        scored.sort(key=lambda item: (-item.final_score, item.image_id))
        return [
            V1RankedResult(
                query_id=pipeline_input.query_id,
                query_text=pipeline_input.query_text,
                rank=index,
                image_id=item.image_id,
                file_path=item.file_path,
                brand=item.brand,
                final_score=item.final_score,
                text_score=item.text_score,
                image_score=item.image_score,
                metadata=item.metadata,
            )
            for index, item in enumerate(scored[: self.config.top_k], start=1)
        ]

    def _compose_final_score(self, text_score: float, image_score: float) -> float:
        if self.config.retrieval_mode == "text_only":
            return text_score
        if self.config.retrieval_mode == "image_only":
            return image_score
        return (
            self.config.text_weight * text_score
            + self.config.image_weight * image_score
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
