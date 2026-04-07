"""Shared concept and attribute helpers for the V3 retrieval pipeline."""

from __future__ import annotations

from collections.abc import Iterable

MULTI_VALUE_SEPARATOR = "|"
COLOR_TOKENS = (
    "charcoal",
    "burgundy",
    "lavender",
    "maroon",
    "silver",
    "purple",
    "yellow",
    "orange",
    "white",
    "black",
    "beige",
    "brown",
    "camel",
    "cream",
    "green",
    "khaki",
    "ivory",
    "olive",
    "navy",
    "blue",
    "pink",
    "gray",
    "grey",
    "gold",
    "red",
    "tan",
    "multicolor",
)
STYLE_CONCEPT_SYNONYMS: dict[str, tuple[str, ...]] = {
    "vintage": (
        "vintage",
        "retro",
        "worn in",
        "worn-in",
        "washed",
        "distressed",
        "aged",
        "70s",
        "1970s",
        "80s",
        "1980s",
        "90s",
        "1990s",
        "2000s revival",
    ),
    "minimal": ("minimal", "clean", "pared back", "pared-back"),
    "romantic": ("romantic", "soft", "delicate"),
    "avant-garde": ("avant garde", "avant-garde"),
}
STYLE_CONCEPT_CANONICALS = tuple(STYLE_CONCEPT_SYNONYMS.keys())
COLOR_STYLE_TAG_HINTS = {
    "two-tone": ("two tone", "two-tone"),
    "multicolor": ("multicolor", "multi color", "multi-color", "colorful"),
    "color-block": ("color block", "color-block", "colorblock"),
}
APPAREL_CATEGORY_HINTS = frozenset(
    {
        "blazer",
        "blouse",
        "bodysuit",
        "cape",
        "cardigan",
        "coat",
        "dress",
        "hoodie",
        "jacket",
        "jeans",
        "jumpsuit",
        "knit",
        "outerwear",
        "pants",
        "poncho",
        "shirt",
        "shorts",
        "skirt",
        "sweater",
        "t shirt",
        "top",
        "trench",
        "trousers",
        "vest",
        "waistcoat",
    }
)


def normalize_token(value: str) -> str:
    return " ".join(value.lower().replace("-", " ").strip().split())


def dedupe_preserve_order(values: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        normalized = normalize_token(value)
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def split_multi_value_text(value: str) -> list[str]:
    cleaned = value.strip()
    if not cleaned:
        return []
    return [item.strip() for item in cleaned.split(MULTI_VALUE_SEPARATOR) if item.strip()]


def extract_style_concepts(text: str) -> list[str]:
    normalized_text = normalize_token(text)
    if not normalized_text:
        return []
    padded_text = f" {normalized_text} "
    concepts: list[str] = []
    for canonical, variants in STYLE_CONCEPT_SYNONYMS.items():
        for variant in variants:
            normalized_variant = normalize_token(variant)
            if not normalized_variant:
                continue
            if f" {normalized_variant} " in padded_text:
                concepts.append(canonical)
                break
    return dedupe_preserve_order(concepts)


def extract_color_style_tags(text: str) -> list[str]:
    normalized_text = normalize_token(text)
    if not normalized_text:
        return []
    padded_text = f" {normalized_text} "
    matched: list[str] = []
    for canonical, variants in COLOR_STYLE_TAG_HINTS.items():
        for variant in variants:
            normalized_variant = normalize_token(variant)
            if f" {normalized_variant} " in padded_text:
                matched.append(canonical)
                break
    return dedupe_preserve_order(matched)


def is_apparel_category(category: str) -> bool:
    return normalize_token(category) in APPAREL_CATEGORY_HINTS
