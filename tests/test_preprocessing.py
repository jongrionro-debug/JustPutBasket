import csv
from pathlib import Path

from switch_query.tagging.preprocessing import (
    CanonicalMappingRow,
    TaggingResult,
    apply_canonical_mappings,
    build_frequency_rows,
    build_full_manifest,
    build_image_inventory,
    build_sample_manifest,
    evaluate_retrieval,
    run_rough_tagging,
    seed_canonical_mappings,
)


class FakeTagger:
    def tag_image(self, sample):
        if sample.brand == "alpha":
            return TaggingResult(
                caption="black wool coat",
                category="coat|trousers",
                silhouette="tailored",
                color="black",
                material="wool",
                pattern="solid",
                texture="smooth",
                mood="minimal",
                season="fall",
                era="modern",
                detail="long wool coat|tailored trousers",
                review_needed=False,
                confidence_note="high",
            )
        return TaggingResult(
            caption="red silk dress",
            category="gown",
            silhouette="fluid",
            color="scarlet",
            material="silk",
            pattern="solid",
            texture="satin",
            mood="romantic",
            season="spring",
            era="vintage",
            detail="evening gown|heels",
            review_needed=True,
            confidence_note="check subtype",
        )


def create_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")


def test_inventory_only_collects_collection_images(tmp_path: Path) -> None:
    create_image(tmp_path / "2026" / "spring-ready-to-wear" / "alpha" / "collection" / "0001_a.jpg")
    create_image(tmp_path / "2026" / "spring-ready-to-wear" / "alpha" / "lookbook" / "0002_b.jpg")
    create_image(tmp_path / "2026" / "spring-ready-to-wear" / "beta" / "collection" / "0001_c.jpg")

    rows = build_image_inventory(tmp_path)

    assert len(rows) == 2
    assert {row.brand for row in rows} == {"alpha", "beta"}
    assert {row.year for row in rows} == {"2026"}
    assert {row.season_group for row in rows} == {"spring-ready-to-wear"}
    assert all(row.source_type == "collection" for row in rows)


def test_sample_manifest_takes_first_and_last_per_brand(tmp_path: Path) -> None:
    for name in ["0001_a.jpg", "0002_b.jpg", "0003_c.jpg"]:
        create_image(tmp_path / "2026" / "spring-ready-to-wear" / "alpha" / "collection" / name)
    for name in ["0004_d.jpg", "0005_e.jpg"]:
        create_image(tmp_path / "2026" / "spring-ready-to-wear" / "beta" / "collection" / name)

    inventory = build_image_inventory(tmp_path)
    manifest = build_sample_manifest(inventory)

    alpha_rows = [row for row in manifest if row.brand == "alpha"]
    beta_rows = [row for row in manifest if row.brand == "beta"]
    assert [row.filename for row in alpha_rows] == ["0001_a.jpg", "0003_c.jpg"]
    assert [row.filename for row in beta_rows] == ["0004_d.jpg", "0005_e.jpg"]


def test_full_manifest_includes_every_inventory_row(tmp_path: Path) -> None:
    for name in ["0001_a.jpg", "0002_b.jpg", "0003_c.jpg"]:
        create_image(tmp_path / "2026" / "spring-ready-to-wear" / "alpha" / "collection" / name)

    inventory = build_image_inventory(tmp_path)
    manifest = build_full_manifest(inventory)

    assert len(manifest) == 3
    assert {row.sample_reason for row in manifest} == {"full_inventory"}
    assert [row.filename for row in manifest] == ["0001_a.jpg", "0002_b.jpg", "0003_c.jpg"]


def test_rough_tagging_and_frequency_tables(tmp_path: Path) -> None:
    for name in ["0001_a.jpg", "0002_b.jpg"]:
        create_image(tmp_path / "2026" / "spring-ready-to-wear" / "alpha" / "collection" / name)
    for name in ["0001_c.jpg", "0002_d.jpg"]:
        create_image(tmp_path / "2026" / "spring-ready-to-wear" / "beta" / "collection" / name)

    inventory = build_image_inventory(tmp_path)
    manifest = build_sample_manifest(inventory)
    raw_rows = run_rough_tagging(manifest, FakeTagger())
    frequencies = build_frequency_rows(raw_rows)

    assert len(raw_rows) == 4
    assert {row.raw_season for row in raw_rows} == {"spring"}
    coat_row = next(row for row in frequencies if row.feature == "category" and row.raw_value == "coat")
    trousers_row = next(row for row in frequencies if row.feature == "category" and row.raw_value == "trousers")
    gown_row = next(row for row in frequencies if row.feature == "category" and row.raw_value == "gown")
    detail_row = next(row for row in frequencies if row.feature == "detail" and row.raw_value == "long wool coat")
    season_row = next(row for row in frequencies if row.feature == "season" and row.raw_value == "spring")
    assert coat_row.count == 2
    assert trousers_row.count == 2
    assert gown_row.count == 2
    assert detail_row.count == 2
    assert season_row.count == 4


def test_seed_and_apply_canonical_mappings_preserve_raw_values(tmp_path: Path) -> None:
    for name in ["0001_a.jpg", "0002_b.jpg"]:
        create_image(tmp_path / "2026" / "spring-ready-to-wear" / "beta" / "collection" / name)

    inventory = build_image_inventory(tmp_path)
    manifest = build_sample_manifest(inventory)
    raw_rows = run_rough_tagging(manifest, FakeTagger())
    frequencies = build_frequency_rows(raw_rows)
    mappings = seed_canonical_mappings(frequencies)

    updated = []
    for row in mappings:
        if row.feature == "category" and row.variant == "gown":
            updated.append(
                CanonicalMappingRow(
                    feature=row.feature,
                    canonical="dress",
                    variant=row.variant,
                    mapping_type="parent_map",
                    notes="collapse subtype for v1",
                    status="approved",
                )
            )
        elif row.feature == "color" and row.variant == "scarlet":
            updated.append(
                CanonicalMappingRow(
                    feature=row.feature,
                    canonical="red",
                    variant=row.variant,
                    mapping_type="synonym",
                    notes="shade normalization",
                    status="approved",
                )
            )
        elif row.feature == "detail" and row.variant == "evening gown":
            updated.append(
                CanonicalMappingRow(
                    feature=row.feature,
                    canonical="gown",
                    variant=row.variant,
                    mapping_type="parent_map",
                    notes="collapse subtype for v1",
                    status="approved",
                )
            )
        else:
            updated.append(row)

    normalized = apply_canonical_mappings(raw_rows, updated)

    assert normalized[0].raw_category == "gown"
    assert normalized[0].canonical_category == "dress"
    assert normalized[0].raw_silhouette == "fluid"
    assert normalized[0].canonical_silhouette == "fluid"
    assert normalized[0].raw_detail == "evening gown|heels"
    assert normalized[0].canonical_detail == "gown|heels"
    assert normalized[0].raw_color == "scarlet"
    assert normalized[0].canonical_color == "red"
    assert normalized[0].canonical_era == "vintage"
    assert normalized[0].raw_season == "spring"
    assert normalized[0].canonical_season == "spring"


def test_retrieval_eval_logs_raw_and_canonical_modes(tmp_path: Path) -> None:
    for brand, names in {
        "alpha": ["0001_a.jpg", "0002_b.jpg"],
        "beta": ["0001_c.jpg", "0002_d.jpg"],
    }.items():
        for name in names:
            create_image(tmp_path / "2026" / "spring-ready-to-wear" / brand / "collection" / name)

    inventory = build_image_inventory(tmp_path)
    manifest = build_sample_manifest(inventory)
    raw_rows = run_rough_tagging(manifest, FakeTagger())
    mappings = [
        CanonicalMappingRow("category", "coat", "coat", "review_needed", "", "draft"),
        CanonicalMappingRow("category", "dress", "gown", "parent_map", "", "approved"),
        CanonicalMappingRow("color", "black", "black", "review_needed", "", "draft"),
        CanonicalMappingRow("color", "red", "scarlet", "synonym", "", "approved"),
        CanonicalMappingRow("material", "wool", "wool", "review_needed", "", "draft"),
        CanonicalMappingRow("material", "silk", "silk", "review_needed", "", "draft"),
        CanonicalMappingRow("mood", "minimal", "minimal", "review_needed", "", "draft"),
        CanonicalMappingRow("mood", "romantic", "romantic", "review_needed", "", "draft"),
    ]
    normalized = apply_canonical_mappings(raw_rows, mappings)
    logs = evaluate_retrieval(raw_rows, normalized)

    assert {row.mode for row in logs} == {"raw", "canonical"}
    assert any(row.query_id.startswith("category_") for row in logs)


def test_sample_manifest_splits_brands_by_year_and_season(tmp_path: Path) -> None:
    create_image(tmp_path / "2025" / "spring-ready-to-wear" / "alpha" / "collection" / "0001_a.jpg")
    create_image(tmp_path / "2025" / "spring-ready-to-wear" / "alpha" / "collection" / "0002_b.jpg")
    create_image(tmp_path / "2026" / "spring-ready-to-wear" / "alpha" / "collection" / "0003_c.jpg")
    create_image(tmp_path / "2026" / "spring-ready-to-wear" / "alpha" / "collection" / "0004_d.jpg")

    inventory = build_image_inventory(tmp_path)
    manifest = build_sample_manifest(inventory)

    alpha_rows = [row for row in manifest if row.brand == "alpha"]

    assert len(alpha_rows) == 4
    assert [row.year for row in alpha_rows] == ["2025", "2025", "2026", "2026"]
