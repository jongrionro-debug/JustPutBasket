"""Image module v1 exports."""

from .attributes import DEFAULT_ATTRIBUTE_NAMES
from .models import (
    FeedbackEvent,
    FeedbackEventType,
    GeneratedImage,
    ImageModuleInput,
    ImageModuleOutput,
    ImageRef,
    ImageRecord,
    RankedImage,
)
from .openai_baseline import (
    BaselineConfig,
    CsvGoogleSheetsStore,
    LocalVectorCache,
    OpenAIBaselineImageModule,
    OpenAIEmbeddingClient,
    OpenAIImageGenerator,
    OpenAIVisionTagger,
)
from .pipeline import ImageModulePipeline, PipelineConfig
from .storage import (
    InMemoryFeedbackStore,
    InMemoryRelationalStore,
    InMemoryVectorStore,
    build_archive_image,
)
from .synonyms import SynonymCatalog

__all__ = [
    "DEFAULT_ATTRIBUTE_NAMES",
    "FeedbackEvent",
    "FeedbackEventType",
    "GeneratedImage",
    "ImageModuleInput",
    "ImageModuleOutput",
    "ImageModulePipeline",
    "ImageRef",
    "BaselineConfig",
    "CsvGoogleSheetsStore",
    "ImageRecord",
    "InMemoryFeedbackStore",
    "InMemoryRelationalStore",
    "InMemoryVectorStore",
    "LocalVectorCache",
    "OpenAIBaselineImageModule",
    "OpenAIEmbeddingClient",
    "OpenAIImageGenerator",
    "OpenAIVisionTagger",
    "PipelineConfig",
    "RankedImage",
    "SynonymCatalog",
    "build_archive_image",
]

from .preprocessing import (
    BlankTagger,
    CanonicalMappingRow,
    FrequencyRow,
    InventoryRow,
    NormalizedTagRow,
    RawTagRow,
    RetrievalEvalRow,
    RetrievalQuery,
    SampleRow,
    SubprocessJsonTagger,
    TaggingResult as PreprocessingTaggingResult,
    apply_canonical_mappings,
    build_default_queries,
    build_frequency_rows,
    build_image_inventory,
    build_sample_manifest,
    evaluate_retrieval,
    read_canonical_mappings,
    run_rough_tagging,
    seed_canonical_mappings,
    write_csv,
)

__all__.extend(
    [
        "BlankTagger",
        "CanonicalMappingRow",
        "FrequencyRow",
        "InventoryRow",
        "NormalizedTagRow",
        "PreprocessingTaggingResult",
        "RawTagRow",
        "RetrievalEvalRow",
        "RetrievalQuery",
        "SampleRow",
        "SubprocessJsonTagger",
        "apply_canonical_mappings",
        "build_default_queries",
        "build_frequency_rows",
        "build_image_inventory",
        "build_sample_manifest",
        "evaluate_retrieval",
        "read_canonical_mappings",
        "run_rough_tagging",
        "seed_canonical_mappings",
        "write_csv",
    ]
)
