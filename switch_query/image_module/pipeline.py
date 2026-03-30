"""Image module v1 prototype pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .attributes import (
    DEFAULT_ATTRIBUTE_NAMES,
    attribute_importance_map,
    balance_bucket,
    stage_attribute_weight,
    stage_similarity_weight,
    synthetic_reference_count,
)
from .models import (
    FeedbackEvent,
    FeedbackEventType,
    GeneratedImage,
    ImageModuleInput,
    ImageModuleOutput,
    ImageRef,
    RankedImage,
)
from .storage import InMemoryFeedbackStore, InMemoryRelationalStore, InMemoryVectorStore
from .synonyms import SynonymCatalog


@dataclass(slots=True)
class PipelineConfig:
    text_weight: float = 0.75
    synthetic_reference_weight: float = 1.10
    uploaded_image_weight: float = 1.35
    top_k: int = 8


class ImageModulePipeline:
    """Prototype image module matching the agreed v1 contract."""

    def __init__(
        self,
        relational_store: InMemoryRelationalStore,
        vector_store: InMemoryVectorStore,
        feedback_store: InMemoryFeedbackStore,
        synonym_catalog: SynonymCatalog,
        config: PipelineConfig | None = None,
    ) -> None:
        self.relational_store = relational_store
        self.vector_store = vector_store
        self.feedback_store = feedback_store
        self.synonym_catalog = synonym_catalog
        self.config = config or PipelineConfig()
        self.term_to_attributes = self._build_term_to_attributes()

    def run(self, module_input: ImageModuleInput) -> ImageModuleOutput:
        requested_attributes = self._extract_requested_attributes(module_input.query_text)
        generated_results = self._generate_synthetic_references(module_input, requested_attributes)

        query_vector = self._compose_query_vector(
            query_text=module_input.query_text,
            requested_attributes=requested_attributes,
            user_uploaded_images=module_input.user_uploaded_images,
            synthetic_references=generated_results,
        )
        archive_results = self._retrieve_and_rerank(
            query_vector=query_vector,
            requested_attributes=requested_attributes,
            stage=module_input.stage,
            balance_score=module_input.balance_score,
        )

        return ImageModuleOutput(
            archive_results=archive_results,
            generated_results=generated_results,
        )

    def record_feedback(
        self,
        event_type: FeedbackEventType,
        target_id: str,
        target_kind: str,
        module_input: ImageModuleInput,
    ) -> None:
        self.feedback_store.record(
            FeedbackEvent(
                event_type=event_type,
                target_id=target_id,
                target_kind=target_kind,  # type: ignore[arg-type]
                query_text=module_input.query_text,
                stage=module_input.stage,
                balance_score=module_input.balance_score,
            )
        )

    def _generate_synthetic_references(
        self,
        module_input: ImageModuleInput,
        requested_attributes: dict[str, str],
    ) -> list[GeneratedImage]:
        count = synthetic_reference_count(module_input.balance_score)
        bucket = balance_bucket(module_input.balance_score)
        results: list[GeneratedImage] = []
        for index in range(count):
            attributes = self._perturb_attributes_for_diversity(
                base_attributes=requested_attributes,
                balance_score=module_input.balance_score,
                variant_index=index,
            )
            results.append(
                GeneratedImage(
                    generated_id=f"gen-{bucket}-{index + 1}",
                    role="synthetic_reference",
                    prompt_summary=self._build_prompt_summary(
                        module_input.query_text, attributes, index
                    ),
                    balance_bucket=bucket,
                    used_for_retrieval=True,
                    attributes=attributes,
                )
            )
        return results

    def _compose_query_vector(
        self,
        query_text: str,
        requested_attributes: dict[str, str],
        user_uploaded_images: list[ImageRef],
        synthetic_references: list[GeneratedImage],
    ) -> dict[str, float]:
        combined: dict[str, float] = {}
        self._merge_weighted_vector(
            combined,
            self._vectorize_attributes(requested_attributes, include_text_bias=True),
            self.config.text_weight,
        )
        for image in user_uploaded_images:
            normalized_attributes = self.synonym_catalog.normalize_attributes(image.attributes)
            self._merge_weighted_vector(
                combined,
                self._vectorize_attributes(normalized_attributes),
                self.config.uploaded_image_weight,
            )
        for image in synthetic_references:
            self._merge_weighted_vector(
                combined,
                self._vectorize_attributes(image.attributes),
                self.config.synthetic_reference_weight,
            )
        return combined

    def _retrieve_and_rerank(
        self,
        query_vector: dict[str, float],
        requested_attributes: dict[str, str],
        stage: str,
        balance_score: float,
    ) -> list[RankedImage]:
        candidates = self.vector_store.search(query_vector, self.config.top_k)
        results: list[RankedImage] = []
        for image_id, embedding_score in candidates:
            image = self.relational_store.get_image(image_id)
            tag_score, matched_attributes, penalized_attributes = self._score_attributes(
                requested_attributes=requested_attributes,
                candidate_attributes=image.attributes,
                stage=stage,
                balance_score=balance_score,
            )
            final_score = (
                embedding_score * stage_similarity_weight(stage) * 0.65 + tag_score * 0.35
            )
            results.append(
                RankedImage(
                    image_id=image_id,
                    source=image.source,
                    final_score=round(final_score, 6),
                    embedding_score=round(embedding_score, 6),
                    tag_score=round(tag_score, 6),
                    matched_attributes=matched_attributes,
                    penalized_attributes=penalized_attributes,
                    metadata=image.metadata,
                )
            )
        return sorted(results, key=lambda result: result.final_score, reverse=True)

    def _score_attributes(
        self,
        requested_attributes: dict[str, str],
        candidate_attributes: dict[str, str],
        stage: str,
        balance_score: float,
    ) -> tuple[float, dict[str, str], dict[str, str]]:
        if not requested_attributes:
            return 0.0, {}, {}

        matched: dict[str, str] = {}
        penalized: dict[str, str] = {}
        total_weight = 0.0
        score = 0.0
        penalty_multiplier = 0.55 if balance_score <= -0.15 else 0.90

        for attribute_name, requested_value in requested_attributes.items():
            weight = stage_attribute_weight(stage, attribute_name)
            total_weight += weight
            candidate_value = candidate_attributes.get(attribute_name)
            if candidate_value == requested_value:
                matched[attribute_name] = requested_value
                score += weight
            elif candidate_value:
                penalized[attribute_name] = candidate_value
                score -= weight * penalty_multiplier

        if total_weight == 0:
            return 0.0, matched, penalized
        return score / total_weight, matched, penalized

    def _extract_requested_attributes(self, query_text: str) -> dict[str, str]:
        normalized_text = self.synonym_catalog._normalize_token(query_text)
        requested: dict[str, str] = {}
        for canonical in self.synonym_catalog.canonical_to_variants:
            if canonical in normalized_text:
                candidate_attributes = self.term_to_attributes.get(canonical, ())
                for attribute_name in candidate_attributes:
                    requested[attribute_name] = canonical
        for attribute_name in DEFAULT_ATTRIBUTE_NAMES:
            marker = f"{attribute_name}:"
            if marker in normalized_text:
                value = normalized_text.split(marker, 1)[1].split(",")[0].strip()
                if value:
                    requested[attribute_name] = self.synonym_catalog.normalize_value(value)
        return requested

    def _perturb_attributes_for_diversity(
        self,
        base_attributes: dict[str, str],
        balance_score: float,
        variant_index: int,
    ) -> dict[str, str]:
        attributes = dict(base_attributes)
        if balance_score > -0.15 or not attributes:
            return attributes

        optional_mood_variants = ("editorial", "romantic", "minimal", "dramatic")
        optional_pattern_variants = ("solid", "tailored", "flowing", "structured")
        if "mood" in attributes:
            attributes["mood"] = optional_mood_variants[variant_index % len(optional_mood_variants)]
        if "pattern" in attributes:
            attributes["pattern"] = optional_pattern_variants[
                variant_index % len(optional_pattern_variants)
            ]
        return self.synonym_catalog.normalize_attributes(attributes)

    def _build_prompt_summary(
        self,
        query_text: str,
        attributes: dict[str, str],
        variant_index: int,
    ) -> str:
        key_fragments = ", ".join(f"{key}={value}" for key, value in sorted(attributes.items()))
        suffix = f" variant {variant_index + 1}" if variant_index else ""
        return f"{query_text.strip()} | {key_fragments}{suffix}".strip()

    def _vectorize_attributes(
        self,
        attributes: dict[str, str],
        include_text_bias: bool = False,
    ) -> dict[str, float]:
        vector: dict[str, float] = {}
        importance_map = attribute_importance_map("mood_board")
        for attribute_name, value in attributes.items():
            vector[f"{attribute_name}:{value}"] = importance_map.get(attribute_name, 1.0)
        if include_text_bias:
            for value in attributes.values():
                vector[f"text:{value}"] = 0.3
        return vector

    @staticmethod
    def _merge_weighted_vector(
        target: dict[str, float],
        vector: dict[str, float],
        weight: float,
    ) -> None:
        for key, value in vector.items():
            target[key] = target.get(key, 0.0) + value * weight

    def _build_term_to_attributes(self) -> dict[str, tuple[str, ...]]:
        mapping: dict[str, set[str]] = {}
        for image in self.relational_store.list_images():
            for attribute_name, value in image.attributes.items():
                normalized_value = self.synonym_catalog.normalize_value(value)
                mapping.setdefault(normalized_value, set()).add(attribute_name)
        return {
            term: tuple(
                sorted(
                    attribute_names,
                    key=lambda attribute_name: DEFAULT_ATTRIBUTE_NAMES.index(attribute_name),
                )
            )
            for term, attribute_names in mapping.items()
        }


def build_pipeline_with_archive(
    archive_images: Iterable[tuple[str, dict[str, str], dict[str, str] | None]],
    synonym_catalog: SynonymCatalog,
    config: PipelineConfig | None = None,
) -> ImageModulePipeline:
    """Helper for quickly building a ready-to-run pipeline in demos/tests."""
    from .storage import build_archive_image

    relational_store = InMemoryRelationalStore()
    vector_store = InMemoryVectorStore()
    feedback_store = InMemoryFeedbackStore()

    for image_id, attributes, metadata in archive_images:
        image = build_archive_image(
            image_id=image_id,
            attributes=attributes,
            synonym_catalog=synonym_catalog,
            metadata=metadata,
        )
        relational_store.add_image(image)
        vector_store.add_vector(image.image_id, image.embedding)

    return ImageModulePipeline(
        relational_store=relational_store,
        vector_store=vector_store,
        feedback_store=feedback_store,
        synonym_catalog=synonym_catalog,
        config=config,
    )
