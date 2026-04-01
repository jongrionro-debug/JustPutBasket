"""V1 image retrieval scaffold."""

from typing import TYPE_CHECKING

from .encoder import DEFAULT_SIGLIP2_MODEL, SigLIP2Encoder, SigLIP2EncoderConfig
from .generator import LUXIA_DALLE3_URL, LuxiaImageGenerator, LuxiaImageGeneratorConfig
from .index import InMemoryArchiveIndex, JsonArchiveIndexStore
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
from .pipeline import V1Pipeline
from .policies import (
    DEFAULT_IMAGE_WEIGHT,
    DEFAULT_TEXT_WEIGHT,
    DEFAULT_TOP_K,
    RetrievalMode,
    V1PipelineConfig,
    average_dense_vectors,
    balance_bucket,
    synthetic_reference_count,
)

if TYPE_CHECKING:
    from .cli import BuildIndexResult, RunQueryResult


def __getattr__(name: str):
    if name in {"build_archive_index", "BuildIndexResult", "run_query", "RunQueryResult"}:
        from . import cli as _cli

        return getattr(_cli, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BuildIndexResult",
    "DEFAULT_SIGLIP2_MODEL",
    "DEFAULT_IMAGE_WEIGHT",
    "DEFAULT_TEXT_WEIGHT",
    "DEFAULT_TOP_K",
    "GeneratedReference",
    "InMemoryArchiveIndex",
    "JsonArchiveIndexStore",
    "LUXIA_DALLE3_URL",
    "LocalIndexStore",
    "LuxiaImageGenerator",
    "LuxiaImageGeneratorConfig",
    "MultimodalEncoder",
    "RetrievalMode",
    "SigLIP2Encoder",
    "SigLIP2EncoderConfig",
    "SyntheticImageGenerator",
    "V1ArchiveRecord",
    "V1Pipeline",
    "V1PipelineConfig",
    "V1PipelineInput",
    "V1PipelineOutput",
    "V1QueryArtifacts",
    "V1RankedResult",
    "V1ScoredCandidate",
    "average_dense_vectors",
    "balance_bucket",
    "build_archive_index",
    "run_query",
    "synthetic_reference_count",
]
