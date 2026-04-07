"""V3 item-aware retrieval package."""

from .batch_probe import LuxiaBatchProbeResult, run_luxia_batch_capability_probe
from .index import build_archive_index
from .models import (
    LocalIndexStore,
    MultimodalEncoder,
    Stage,
    V3ArchiveDocument,
    V3ArchiveIndex,
    V3CandidateMode,
    V3CandidateTrace,
    V3DocumentItem,
    V3EmbeddingCandidate,
    V3IndexedDocument,
    V3ItemExtractionInput,
    V3ItemExtractionOutput,
    V3ItemAssignment,
    V3ItemMatchStatus,
    V3ParsedQuery,
    V3PipelineInput,
    V3PipelineOutput,
    V3RankedResult,
    V3TargetItem,
)
from .item_extractor import (
    GeminiItemExtractor,
    GeminiItemExtractorConfig,
    LuxiaItemExtractor,
    LuxiaItemExtractorConfig,
)
from .llm_parser import LUXIA_CHAT_URL, LuxiaV3QueryParser, LuxiaV3QueryParserConfig
from .pipeline import V3Pipeline, V3PipelineConfig
from .preprocessing import backfill_archive_document_style_tags, read_archive_documents_jsonl
from .query_validation import V3QueryValidationResult, ensure_valid_v3_parsed_query, validate_v3_parsed_query
from .ranker import V3Ranker, V3RankerConfig
from .retriever import V3Retriever, V3RetrieverConfig, dense_cosine_similarity
from .storage import InMemoryArchiveIndex, JsonArchiveIndexStore

__all__ = [
    "GeminiItemExtractor",
    "GeminiItemExtractorConfig",
    "InMemoryArchiveIndex",
    "JsonArchiveIndexStore",
    "LUXIA_CHAT_URL",
    "LocalIndexStore",
    "LuxiaBatchProbeResult",
    "LuxiaItemExtractor",
    "LuxiaItemExtractorConfig",
    "LuxiaV3QueryParser",
    "LuxiaV3QueryParserConfig",
    "MultimodalEncoder",
    "Stage",
    "V3ArchiveDocument",
    "V3ArchiveIndex",
    "V3CandidateMode",
    "V3CandidateTrace",
    "V3DocumentItem",
    "V3EmbeddingCandidate",
    "V3IndexedDocument",
    "V3ItemAssignment",
    "V3ItemExtractionInput",
    "V3ItemExtractionOutput",
    "V3ItemMatchStatus",
    "V3ParsedQuery",
    "V3Pipeline",
    "V3PipelineConfig",
    "V3PipelineInput",
    "V3PipelineOutput",
    "V3QueryValidationResult",
    "V3RankedResult",
    "V3Ranker",
    "V3RankerConfig",
    "V3Retriever",
    "V3RetrieverConfig",
    "V3TargetItem",
    "build_archive_index",
    "backfill_archive_document_style_tags",
    "dense_cosine_similarity",
    "ensure_valid_v3_parsed_query",
    "read_archive_documents_jsonl",
    "run_luxia_batch_capability_probe",
    "validate_v3_parsed_query",
]
