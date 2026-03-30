import csv
import json
from pathlib import Path

from switch_query.image_module import (
    BaselineConfig,
    CsvGoogleSheetsStore,
    ImageModuleInput,
    ImageRef,
    LocalVectorCache,
    OpenAIBaselineImageModule,
    GeneratedImage,
    SynonymCatalog,
)
from switch_query.image_module.openai_baseline import TaggingResult


class FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    1.0 if "coat" in lowered else 0.0,
                    1.0 if "black" in lowered else 0.0,
                    1.0 if "dress" in lowered else 0.0,
                    1.0 if "silver" in lowered else 0.0,
                    1.0 if "tailored" in lowered else 0.0,
                    1.0 if "editorial" in lowered else 0.0,
                ]
            )
        return vectors


class FakeImageGenerator:
    def generate(self, prompt: str, count: int) -> list[str]:
        return ["ZmFrZV9pbWFnZQ==" for _ in range(count)]


class FakeVisionTagger:
    def __init__(self, mapping: dict[str, TaggingResult]) -> None:
        self.mapping = mapping

    def tag_image(self, image: ImageRef, query_text: str, stage: str) -> TaggingResult:
        return self.mapping[image.image_id]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def build_module(tmp_path: Path) -> OpenAIBaselineImageModule:
    archive_csv = tmp_path / "archive.csv"
    synonyms_csv = tmp_path / "synonyms.csv"
    feedback_csv = tmp_path / "feedback.csv"
    vector_cache_path = tmp_path / "vector_cache.json"

    write_csv(
        archive_csv,
        [
            "image_id",
            "source",
            "title",
            "category",
            "silhouette",
            "color",
            "material",
            "mood",
            "detail",
        ],
        [
            [
                "look-1",
                "vogue_runway",
                "Structured Black Coat",
                "coat",
                "tailored",
                "black",
                "wool",
                "minimal",
                "structured",
            ],
            [
                "look-2",
                "vogue_runway",
                "Fluid Silver Dress",
                "dress",
                "fluid",
                "silver",
                "silk",
                "editorial",
                "draped",
            ],
        ],
    )
    write_csv(
        synonyms_csv,
        ["canonical", "variants"],
        [
            ["coat", "outerwear"],
            ["tailored", "sharp tailored"],
            ["black", "jet black"],
            ["editorial", "fashion editorial"],
        ],
    )

    config = BaselineConfig(
        archive_csv_path=str(archive_csv),
        synonyms_csv_path=str(synonyms_csv),
        feedback_csv_path=str(feedback_csv),
        vector_cache_path=str(vector_cache_path),
        top_k=2,
    )
    store = CsvGoogleSheetsStore(
        archive_csv_path=str(archive_csv),
        synonyms_csv_path=str(synonyms_csv),
        feedback_csv_path=str(feedback_csv),
    )
    synonym_catalog = store.load_synonym_catalog()
    vector_cache = LocalVectorCache.load(str(vector_cache_path))
    module = OpenAIBaselineImageModule(
        store=store,
        vector_cache=vector_cache,
        synonym_catalog=synonym_catalog,
        embedding_client=FakeEmbeddingClient(),
        image_generator=FakeImageGenerator(),
        vision_tagger=FakeVisionTagger(
            {
                "user-1": TaggingResult(
                    attributes={
                        "category": "coat",
                        "silhouette": "tailored",
                        "color": "black",
                    },
                    caption="black tailored coat",
                ),
                "gen-convergent-1": TaggingResult(
                    attributes={
                        "category": "coat",
                        "silhouette": "tailored",
                        "color": "black",
                    },
                    caption="synthetic black tailored coat",
                ),
                "gen-divergent-1": TaggingResult(
                    attributes={"category": "coat", "color": "black"},
                    caption="synthetic coat",
                ),
                "gen-divergent-2": TaggingResult(
                    attributes={"category": "coat", "color": "black"},
                    caption="synthetic coat",
                ),
                "gen-divergent-3": TaggingResult(
                    attributes={"category": "coat", "color": "black"},
                    caption="synthetic coat",
                ),
                "gen-divergent-4": TaggingResult(
                    attributes={"category": "coat", "color": "black"},
                    caption="synthetic coat",
                ),
            }
        ),
        config=config,
    )
    return module


def test_build_archive_index_creates_local_vector_cache(tmp_path: Path) -> None:
    module = build_module(tmp_path)

    module.build_archive_index()

    cache = json.loads((tmp_path / "vector_cache.json").read_text(encoding="utf-8"))
    assert set(cache["vectors"]) == {"look-1", "look-2"}


def test_baseline_pipeline_prefers_user_and_synthetic_coat_signal(tmp_path: Path) -> None:
    module = build_module(tmp_path)
    local_image = tmp_path / "user.png"
    local_image.write_bytes(b"fake")
    module_input = ImageModuleInput(
        query_text="Need a black tailored coat for sketch refinement",
        user_uploaded_images=[
            ImageRef(image_id="user-1", local_path=str(local_image), attributes={})
        ],
        stage="sketch_stage",
        balance_score=0.7,
    )

    output = module.run(module_input)

    assert output.archive_results[0].image_id == "look-1"
    assert len(output.generated_results) == 1
    assert isinstance(output.generated_results[0], GeneratedImage)
    assert output.generated_results[0].attributes["category"] == "coat"


def test_divergent_balance_generates_four_synthetic_references(tmp_path: Path) -> None:
    module = build_module(tmp_path)
    module_input = ImageModuleInput(
        query_text="Explore more black coat directions",
        user_uploaded_images=[],
        stage="mood_board",
        balance_score=-0.8,
    )

    output = module.run(module_input)

    assert len(output.generated_results) == 4
    assert all(image.used_for_retrieval for image in output.generated_results)


def test_feedback_is_appended_to_csv(tmp_path: Path) -> None:
    module = build_module(tmp_path)
    module_input = ImageModuleInput(
        query_text="Need black coat directions",
        user_uploaded_images=[],
        stage="mood_board",
        balance_score=-0.4,
    )

    module.record_feedback("save", "look-1", "archive", module_input)

    feedback_rows = list(csv.DictReader(open(tmp_path / "feedback.csv", encoding="utf-8")))
    assert feedback_rows[0]["event_type"] == "save"
    assert feedback_rows[0]["target_id"] == "look-1"
