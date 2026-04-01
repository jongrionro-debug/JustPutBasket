"""OpenAI-backed image module baseline with CSV sheet storage and local vector cache."""

from __future__ import annotations

import base64
import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib import error, request

from switch_query.tagging.attributes import DEFAULT_ATTRIBUTE_NAMES
from switch_query.tagging.synonyms import SynonymCatalog

from .attributes import (
    balance_bucket,
    stage_attribute_weight,
    stage_similarity_weight,
    synthetic_reference_count,
)
from .models import (
    FeedbackEventType,
    GeneratedImage,
    ImageModuleInput,
    ImageModuleOutput,
    ImageRecord,
    ImageRef,
    RankedImage,
)


@dataclass(slots=True)
class BaselineConfig:
    archive_csv_path: str
    synonyms_csv_path: str
    feedback_csv_path: str
    vector_cache_path: str
    generation_model: str = "gpt-image-1.5"
    vision_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-large"
    top_k: int = 8
    text_weight: float = 0.75
    synthetic_reference_weight: float = 1.10
    uploaded_image_weight: float = 1.35
    image_size: str = "1024x1024"
    image_quality: str = "medium"


@dataclass(slots=True)
class ArchiveDocument:
    image_id: str
    summary_text: str
    vector: list[float]


@dataclass(slots=True)
class TaggingResult:
    attributes: dict[str, str]
    caption: str


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return dense embeddings for the provided texts."""


class SyntheticImageGenerator(Protocol):
    def generate(self, prompt: str, count: int) -> list[str]:
        """Return base64-encoded images."""


class VisionTagger(Protocol):
    def tag_image(self, image: ImageRef, query_text: str, stage: str) -> TaggingResult:
        """Produce normalized attributes and caption for a given image."""


class OpenAIHTTPTransport:
    """Tiny HTTP client so the baseline works without extra runtime dependencies."""

    def __init__(self, api_key: str, timeout_seconds: float = 60.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    def post_json(self, url: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        http_request = request.Request(url=url, method="POST", data=body, headers=headers)
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI request failed: {exc.code} {detail}") from exc


class OpenAIEmbeddingClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-large",
        transport: OpenAIHTTPTransport | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddingClient")
        self.model = model
        self.transport = transport or OpenAIHTTPTransport(api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.transport.post_json(
            "https://api.openai.com/v1/embeddings",
            {"model": self.model, "input": texts},
        )
        data = response.get("data", [])
        embeddings = [item["embedding"] for item in data if "embedding" in item]
        if len(embeddings) != len(texts):
            raise RuntimeError("Embedding response size did not match input size")
        return embeddings


class OpenAIImageGenerator:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-image-1.5",
        size: str = "1024x1024",
        quality: str = "medium",
        transport: OpenAIHTTPTransport | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIImageGenerator")
        self.model = model
        self.size = size
        self.quality = quality
        self.transport = transport or OpenAIHTTPTransport(api_key)

    def generate(self, prompt: str, count: int) -> list[str]:
        response = self.transport.post_json(
            "https://api.openai.com/v1/images/generations",
            {
                "model": self.model,
                "prompt": prompt,
                "n": count,
                "size": self.size,
                "quality": self.quality,
            },
        )
        images = [item["b64_json"] for item in response.get("data", []) if "b64_json" in item]
        if len(images) != count:
            raise RuntimeError("Image generation response size did not match request size")
        return images


class OpenAIVisionTagger:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4.1-mini",
        transport: OpenAIHTTPTransport | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIVisionTagger")
        self.model = model
        self.transport = transport or OpenAIHTTPTransport(api_key)

    def tag_image(self, image: ImageRef, query_text: str, stage: str) -> TaggingResult:
        image_part = self._build_image_part(image)
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are a fashion image tagger. Return JSON only. "
                                "Create a short caption plus attributes for these keys only: "
                                f"{', '.join(DEFAULT_ATTRIBUTE_NAMES)}. "
                                "Omit keys when uncertain. Keep values concise."
                            ),
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                f"Query context: {query_text}\n"
                                f"Design stage: {stage}\n"
                                "Return JSON with keys `caption` and `attributes`."
                            ),
                        },
                        image_part,
                    ],
                },
            ],
            "text": {"format": {"type": "json_object"}},
        }
        response = self.transport.post_json("https://api.openai.com/v1/responses", payload)
        parsed = json.loads(self._extract_output_text(response))
        attributes = {
            key: str(value)
            for key, value in parsed.get("attributes", {}).items()
            if key in DEFAULT_ATTRIBUTE_NAMES and value
        }
        caption = str(parsed.get("caption", "")).strip()
        return TaggingResult(attributes=attributes, caption=caption)

    def _build_image_part(self, image: ImageRef) -> dict[str, str]:
        if image.image_url:
            return {"type": "input_image", "image_url": image.image_url, "detail": "low"}
        if image.local_path:
            path = Path(image.local_path)
            encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
            suffix = path.suffix.lower()
            mime_type = "image/png" if suffix == ".png" else "image/jpeg"
            return {
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{encoded}",
                "detail": "low",
            }
        raise ValueError("ImageRef requires image_url or local_path for OpenAIVisionTagger")

    @staticmethod
    def _extract_output_text(response: dict[str, object]) -> str:
        output_text = response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        chunks: list[str] = []
        for item in response.get("output", []):
            if not isinstance(item, dict):
                continue
            for content in item.get("content", []):
                if not isinstance(content, dict):
                    continue
                text = content.get("text")
                if isinstance(text, str):
                    chunks.append(text)
        if not chunks:
            raise RuntimeError("No text content returned by vision response")
        return "".join(chunks)


@dataclass(slots=True)
class CsvGoogleSheetsStore:
    """CSV-backed store shaped to round-trip with Google Sheets tabs."""

    archive_csv_path: str
    synonyms_csv_path: str
    feedback_csv_path: str

    def load_archive_records(self) -> list[ImageRecord]:
        with open(self.archive_csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            records: list[ImageRecord] = []
            for row in reader:
                attributes = {
                    key: row[key].strip()
                    for key in DEFAULT_ATTRIBUTE_NAMES
                    if row.get(key, "").strip()
                }
                metadata = {
                    key: value.strip()
                    for key, value in row.items()
                    if key not in {"image_id", "source", *DEFAULT_ATTRIBUTE_NAMES} and value
                }
                records.append(
                    ImageRecord(
                        image_id=row["image_id"].strip(),
                        source=row.get("source", "vogue_runway").strip() or "vogue_runway",
                        attributes=attributes,
                        embedding={},
                        metadata=metadata,
                    )
                )
        return records

    def load_synonym_catalog(self) -> SynonymCatalog:
        mapping: dict[str, set[str]] = {}
        with open(self.synonyms_csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                canonical = row["canonical"].strip()
                variants = {
                    variant.strip()
                    for variant in row.get("variants", "").split("|")
                    if variant.strip()
                }
                mapping[canonical] = variants
        return SynonymCatalog(mapping)

    def append_feedback(
        self,
        event_type: FeedbackEventType,
        target_id: str,
        target_kind: str,
        module_input: ImageModuleInput,
    ) -> None:
        path = Path(self.feedback_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = path.exists()
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "event_type",
                    "target_id",
                    "target_kind",
                    "query_text",
                    "stage",
                    "balance_score",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(
                {
                    "event_type": event_type,
                    "target_id": target_id,
                    "target_kind": target_kind,
                    "query_text": module_input.query_text,
                    "stage": module_input.stage,
                    "balance_score": module_input.balance_score,
                }
            )


@dataclass(slots=True)
class LocalVectorCache:
    cache_path: str
    vectors: dict[str, list[float]] = field(default_factory=dict)
    payloads: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, cache_path: str) -> "LocalVectorCache":
        path = Path(cache_path)
        if not path.exists():
            return cls(cache_path=cache_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            cache_path=cache_path,
            vectors={key: list(value) for key, value in data.get("vectors", {}).items()},
            payloads=dict(data.get("payloads", {})),
        )

    def get(self, key: str, payload: str) -> list[float] | None:
        if self.payloads.get(key) != payload:
            return None
        return self.vectors.get(key)

    def set(self, key: str, payload: str, vector: list[float]) -> None:
        self.payloads[key] = payload
        self.vectors[key] = vector

    def save(self) -> None:
        path = Path(self.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"vectors": self.vectors, "payloads": self.payloads}, indent=2),
            encoding="utf-8",
        )

    def search(self, query_vector: list[float], limit: int) -> list[tuple[str, float]]:
        scored = [
            (image_id, dense_cosine_similarity(query_vector, vector))
            for image_id, vector in self.vectors.items()
        ]
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]


class OpenAIBaselineImageModule:
    """Baseline pipeline using OpenAI services and a CSV/Sheets-compatible backend."""

    def __init__(
        self,
        store: CsvGoogleSheetsStore,
        vector_cache: LocalVectorCache,
        synonym_catalog: SynonymCatalog,
        embedding_client: EmbeddingClient,
        image_generator: SyntheticImageGenerator,
        vision_tagger: VisionTagger,
        config: BaselineConfig,
    ) -> None:
        self.store = store
        self.vector_cache = vector_cache
        self.synonym_catalog = synonym_catalog
        self.embedding_client = embedding_client
        self.image_generator = image_generator
        self.vision_tagger = vision_tagger
        self.config = config
        self.archive_records = {record.image_id: record for record in self.store.load_archive_records()}

    @classmethod
    def from_config(cls, config: BaselineConfig) -> "OpenAIBaselineImageModule":
        store = CsvGoogleSheetsStore(
            archive_csv_path=config.archive_csv_path,
            synonyms_csv_path=config.synonyms_csv_path,
            feedback_csv_path=config.feedback_csv_path,
        )
        synonym_catalog = store.load_synonym_catalog()
        vector_cache = LocalVectorCache.load(config.vector_cache_path)
        embedding_client = OpenAIEmbeddingClient(model=config.embedding_model)
        image_generator = OpenAIImageGenerator(
            model=config.generation_model,
            size=config.image_size,
            quality=config.image_quality,
        )
        vision_tagger = OpenAIVisionTagger(model=config.vision_model)
        return cls(
            store=store,
            vector_cache=vector_cache,
            synonym_catalog=synonym_catalog,
            embedding_client=embedding_client,
            image_generator=image_generator,
            vision_tagger=vision_tagger,
            config=config,
        )

    def build_archive_index(self, force_refresh: bool = False) -> None:
        documents_to_embed: list[ArchiveDocument] = []
        for record in self.archive_records.values():
            summary = self._record_to_summary(record)
            cached = None if force_refresh else self.vector_cache.get(record.image_id, summary)
            if cached is None:
                documents_to_embed.append(
                    ArchiveDocument(image_id=record.image_id, summary_text=summary, vector=[])
                )

        if documents_to_embed:
            vectors = self.embedding_client.embed_texts(
                [document.summary_text for document in documents_to_embed]
            )
            for document, vector in zip(documents_to_embed, vectors, strict=True):
                self.vector_cache.set(document.image_id, document.summary_text, vector)
            self.vector_cache.save()

    def run(self, module_input: ImageModuleInput) -> ImageModuleOutput:
        self.build_archive_index()

        synthetic_count = synthetic_reference_count(module_input.balance_score)
        generated_images = self._generate_synthetic_images(module_input, synthetic_count)

        user_tagged = [
            self._merge_attributes(image.attributes, self.vision_tagger.tag_image(image, module_input.query_text, module_input.stage))
            for image in module_input.user_uploaded_images
        ]
        synthetic_tagged = [
            self._merge_attributes(image.attributes, self.vision_tagger.tag_image(self._generated_to_ref(image), module_input.query_text, module_input.stage))
            for image in generated_images
        ]

        query_vector = self._compose_query_vector(
            module_input=module_input,
            user_tagged=user_tagged,
            synthetic_tagged=synthetic_tagged,
        )

        requested_attributes = self._requested_attributes_from_tags(
            query_text=module_input.query_text,
            tagged_sets=[result.attributes for result in user_tagged + synthetic_tagged],
        )

        archive_results = self._retrieve_and_rerank(
            query_vector=query_vector,
            requested_attributes=requested_attributes,
            stage=module_input.stage,
            balance_score=module_input.balance_score,
        )

        for generated, tagging in zip(generated_images, synthetic_tagged, strict=True):
            generated.attributes = tagging.attributes

        return ImageModuleOutput(
            archive_results=archive_results,
            generated_results=generated_images,
        )

    def record_feedback(
        self,
        event_type: FeedbackEventType,
        target_id: str,
        target_kind: str,
        module_input: ImageModuleInput,
    ) -> None:
        self.store.append_feedback(event_type, target_id, target_kind, module_input)

    def _generate_synthetic_images(
        self,
        module_input: ImageModuleInput,
        count: int,
    ) -> list[GeneratedImage]:
        prompt = self._build_synthetic_prompt(module_input)
        images = self.image_generator.generate(prompt, count)
        bucket = balance_bucket(module_input.balance_score)
        return [
            GeneratedImage(
                generated_id=f"gen-{bucket}-{index + 1}",
                role="synthetic_reference",
                prompt_summary=prompt,
                balance_bucket=bucket,
                used_for_retrieval=True,
                attributes={},
                image_base64=image_b64,
            )
            for index, image_b64 in enumerate(images)
        ]

    def _build_synthetic_prompt(self, module_input: ImageModuleInput) -> str:
        direction = (
            "Produce diverse fashion directions and vary mood, styling, and silhouette."
            if module_input.balance_score <= -0.15
            else "Produce a tightly aligned fashion reference faithful to the brief."
        )
        stage_hint = (
            "Prioritize atmosphere, color, and inspiration."
            if module_input.stage == "mood_board"
            else "Prioritize garment structure, materials, and construction details."
        )
        return f"{module_input.query_text}\n{direction}\n{stage_hint}"

    def _compose_query_vector(
        self,
        module_input: ImageModuleInput,
        user_tagged: list[TaggingResult],
        synthetic_tagged: list[TaggingResult],
    ) -> list[float]:
        texts = [self._query_text_summary(module_input.query_text)]
        texts.extend(self._tagging_to_summary(tagging) for tagging in user_tagged)
        texts.extend(self._tagging_to_summary(tagging) for tagging in synthetic_tagged)
        vectors = self.embedding_client.embed_texts(texts)

        combined = [0.0] * len(vectors[0])
        self._add_dense_vector(combined, vectors[0], self.config.text_weight)

        offset = 1
        for _tagging in user_tagged:
            self._add_dense_vector(combined, vectors[offset], self.config.uploaded_image_weight)
            offset += 1
        for _tagging in synthetic_tagged:
            self._add_dense_vector(
                combined, vectors[offset], self.config.synthetic_reference_weight
            )
            offset += 1
        return combined

    def _retrieve_and_rerank(
        self,
        query_vector: list[float],
        requested_attributes: dict[str, str],
        stage: str,
        balance_score: float,
    ) -> list[RankedImage]:
        results: list[RankedImage] = []
        for image_id, embedding_score in self.vector_cache.search(query_vector, self.config.top_k):
            record = self.archive_records[image_id]
            tag_score, matched, penalized = self._score_attributes(
                requested_attributes=requested_attributes,
                candidate_attributes=record.attributes,
                stage=stage,
                balance_score=balance_score,
            )
            final_score = (
                embedding_score * stage_similarity_weight(stage) * 0.65 + tag_score * 0.35
            )
            results.append(
                RankedImage(
                    image_id=record.image_id,
                    source=record.source,
                    final_score=round(final_score, 6),
                    embedding_score=round(embedding_score, 6),
                    tag_score=round(tag_score, 6),
                    matched_attributes=matched,
                    penalized_attributes=penalized,
                    metadata=record.metadata,
                )
            )
        return sorted(results, key=lambda item: item.final_score, reverse=True)

    def _requested_attributes_from_tags(
        self,
        query_text: str,
        tagged_sets: list[dict[str, str]],
    ) -> dict[str, str]:
        requested: dict[str, str] = {}
        normalized_query = self.synonym_catalog._normalize_token(query_text)
        for tagged in tagged_sets:
            for key, value in tagged.items():
                normalized = self.synonym_catalog.normalize_value(value)
                if normalized and (normalized in normalized_query or key not in requested):
                    requested[key] = normalized
        return requested

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
            candidate_value = self.synonym_catalog.normalize_value(
                candidate_attributes.get(attribute_name, "")
            )
            if candidate_value == requested_value:
                matched[attribute_name] = requested_value
                score += weight
            elif candidate_value:
                penalized[attribute_name] = candidate_value
                score -= weight * penalty_multiplier

        return (score / total_weight if total_weight else 0.0), matched, penalized

    def _record_to_summary(self, record: ImageRecord) -> str:
        metadata_parts = [value for value in record.metadata.values() if value]
        attribute_parts = [f"{key}: {value}" for key, value in sorted(record.attributes.items())]
        return " | ".join(metadata_parts + attribute_parts)

    def _tagging_to_summary(self, tagging: TaggingResult) -> str:
        parts = [tagging.caption] if tagging.caption else []
        parts.extend(f"{key}: {value}" for key, value in sorted(tagging.attributes.items()))
        return " | ".join(part for part in parts if part)

    def _query_text_summary(self, query_text: str) -> str:
        return query_text.strip()

    def _generated_to_ref(self, image: GeneratedImage) -> ImageRef:
        return ImageRef(
            image_id=image.generated_id,
            description=image.prompt_summary,
            local_path=self._write_temp_generated_image(image),
        )

    def _write_temp_generated_image(self, image: GeneratedImage) -> str:
        if not image.image_base64:
            raise ValueError("Generated image does not contain image_base64 data")
        temp_dir = Path(self.config.vector_cache_path).parent / "generated_refs"
        temp_dir.mkdir(parents=True, exist_ok=True)
        path = temp_dir / f"{image.generated_id}.png"
        path.write_bytes(base64.b64decode(image.image_base64))
        return str(path)

    def _merge_attributes(self, existing: dict[str, str], tagging: TaggingResult) -> TaggingResult:
        merged = dict(tagging.attributes)
        for key, value in existing.items():
            merged[key] = self.synonym_catalog.normalize_value(value)
        return TaggingResult(attributes=merged, caption=tagging.caption)

    @staticmethod
    def _add_dense_vector(target: list[float], vector: list[float], weight: float) -> None:
        for index, value in enumerate(vector):
            target[index] += value * weight


def dense_cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    numerator = sum(lval * rval for lval, rval in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)
