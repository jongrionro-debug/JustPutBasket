from switch_query.image_module import (
    ImageModuleInput,
    ImageModulePipeline,
    ImageRef,
    InMemoryFeedbackStore,
    InMemoryRelationalStore,
    InMemoryVectorStore,
    SynonymCatalog,
    build_archive_image,
)


def build_pipeline() -> ImageModulePipeline:
    synonyms = SynonymCatalog(
        {
            "black": {"jet black"},
            "tailored": {"sharp tailored"},
            "wool": {"wool blend"},
            "minimal": {"clean minimal"},
            "editorial": {"fashion editorial"},
            "structured": {"architectural"},
            "fluid": {"flowing"},
            "dramatic": {"bold dramatic"},
            "dress": {"gown"},
            "coat": {"outerwear"},
            "spring": {"spring summer"},
            "silver": {"metallic silver"},
        }
    )
    relational = InMemoryRelationalStore()
    vector = InMemoryVectorStore()
    feedback = InMemoryFeedbackStore()
    archive_items = [
        build_archive_image(
            "look-1",
            {
                "category": "coat",
                "silhouette": "tailored",
                "color": "black",
                "material": "wool",
                "mood": "minimal",
                "detail": "structured",
                "season": "spring",
            },
            synonyms,
            metadata={"title": "Structured Black Coat"},
        ),
        build_archive_image(
            "look-2",
            {
                "category": "dress",
                "silhouette": "fluid",
                "color": "silver",
                "material": "silk",
                "mood": "editorial",
                "detail": "draped",
                "season": "spring",
            },
            synonyms,
            metadata={"title": "Fluid Silver Dress"},
        ),
        build_archive_image(
            "look-3",
            {
                "category": "coat",
                "silhouette": "tailored",
                "color": "cream",
                "material": "cotton",
                "mood": "romantic",
                "detail": "soft",
                "season": "fall",
            },
            synonyms,
            metadata={"title": "Cream Transitional Coat"},
        ),
    ]
    for image in archive_items:
        relational.add_image(image)
        vector.add_vector(image.image_id, image.embedding)

    return ImageModulePipeline(relational, vector, feedback, synonyms)


def test_user_uploaded_image_remains_anchor_when_synthetic_refs_added() -> None:
    pipeline = build_pipeline()
    module_input = ImageModuleInput(
        query_text=(
            "Need a sharp tailored black wool coat with clean minimal mood and "
            "structured detail for a sketch review"
        ),
        user_uploaded_images=[
            ImageRef(
                image_id="user-1",
                description="reference board pick",
                attributes={
                    "category": "coat",
                    "silhouette": "tailored",
                    "color": "black",
                    "material": "wool",
                    "mood": "minimal",
                    "detail": "structured",
                },
            )
        ],
        stage="sketch_stage",
        balance_score=0.8,
    )

    output = pipeline.run(module_input)

    assert output.archive_results[0].image_id == "look-1"
    assert output.generated_results[0].used_for_retrieval is True


def test_pipeline_works_with_synthetic_reference_only() -> None:
    pipeline = build_pipeline()
    module_input = ImageModuleInput(
        query_text="Looking for a silver editorial gown with flowing silhouette for spring",
        user_uploaded_images=[],
        stage="mood_board",
        balance_score=0.1,
    )

    output = pipeline.run(module_input)

    assert output.archive_results
    assert output.archive_results[0].image_id == "look-2"
    assert len(output.generated_results) == 1


def test_divergent_balance_emits_multiple_synthetic_references() -> None:
    pipeline = build_pipeline()
    module_input = ImageModuleInput(
        query_text="Explore black tailored coat directions with minimal fashion editorial mood",
        user_uploaded_images=[],
        stage="mood_board",
        balance_score=-0.9,
    )

    output = pipeline.run(module_input)

    assert len(output.generated_results) == 4
    assert all(result.used_for_retrieval for result in output.generated_results)


def test_convergent_balance_emits_single_reference() -> None:
    pipeline = build_pipeline()
    module_input = ImageModuleInput(
        query_text="Refine black tailored wool coat with structured details",
        user_uploaded_images=[],
        stage="sketch_stage",
        balance_score=0.6,
    )

    output = pipeline.run(module_input)

    assert len(output.generated_results) == 1


def test_synthetic_references_are_returned_in_generated_section() -> None:
    pipeline = build_pipeline()
    module_input = ImageModuleInput(
        query_text="Need minimal black coat references",
        user_uploaded_images=[],
        stage="mood_board",
        balance_score=-0.3,
    )

    output = pipeline.run(module_input)

    assert {result.role for result in output.generated_results} == {"synthetic_reference"}
    assert all(result.used_for_retrieval for result in output.generated_results)


def test_tag_reranking_penalizes_attribute_mismatch() -> None:
    pipeline = build_pipeline()
    module_input = ImageModuleInput(
        query_text="black tailored coat wool minimal structured",
        user_uploaded_images=[],
        stage="sketch_stage",
        balance_score=0.7,
    )

    output = pipeline.run(module_input)

    top_result = output.archive_results[0]
    second_result = output.archive_results[1]
    assert top_result.image_id == "look-1"
    assert second_result.image_id != "look-1"
    assert second_result.penalized_attributes
    assert top_result.final_score > second_result.final_score


def test_feedback_events_are_stored_for_archive_and_generated_results() -> None:
    pipeline = build_pipeline()
    module_input = ImageModuleInput(
        query_text="Need black coat directions",
        user_uploaded_images=[],
        stage="mood_board",
        balance_score=-0.8,
    )

    output = pipeline.run(module_input)
    pipeline.record_feedback("select", output.archive_results[0].image_id, "archive", module_input)
    pipeline.record_feedback(
        "save", output.generated_results[0].generated_id, "generated", module_input
    )
    pipeline.record_feedback(
        "exclude", output.generated_results[1].generated_id, "generated", module_input
    )

    assert [event.event_type for event in pipeline.feedback_store.events] == [
        "select",
        "save",
        "exclude",
    ]
    assert {event.target_kind for event in pipeline.feedback_store.events} == {
        "archive",
        "generated",
    }
