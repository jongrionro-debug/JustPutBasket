"""Canonical vocabulary handling for user and archive tags."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SynonymCatalog:
    canonical_to_variants: dict[str, set[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, set[str]] = {}
        for canonical, variants in self.canonical_to_variants.items():
            key = self._normalize_token(canonical)
            values = {self._normalize_token(canonical)}
            values.update(self._normalize_token(variant) for variant in variants)
            normalized[key] = values
        self.canonical_to_variants = normalized

    def normalize_value(self, value: str) -> str:
        token = self._normalize_token(value)
        for canonical, variants in self.canonical_to_variants.items():
            if token in variants:
                return canonical
        return token

    def normalize_attributes(self, attributes: dict[str, str]) -> dict[str, str]:
        return {name: self.normalize_value(value) for name, value in attributes.items() if value}

    def canonical_terms(self) -> set[str]:
        terms: set[str] = set()
        for canonical, variants in self.canonical_to_variants.items():
            terms.add(canonical)
            terms.update(variants)
        return terms

    @staticmethod
    def _normalize_token(value: str) -> str:
        return " ".join(value.lower().strip().replace("-", " ").split())
