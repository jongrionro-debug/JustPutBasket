"""Storage adapters for V2 documents and archive indexes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import V2ArchiveIndex, V2IndexedDocument


@dataclass(slots=True)
class InMemoryArchiveIndex:
    index: V2ArchiveIndex = field(default_factory=V2ArchiveIndex)

    def save(self, index: V2ArchiveIndex) -> None:
        self.index = index

    def load(self) -> V2ArchiveIndex:
        return V2ArchiveIndex(
            documents=list(self.index.documents),
            feature_vocabulary={feature: dict(values) for feature, values in self.index.feature_vocabulary.items()},
        )


@dataclass(slots=True)
class JsonArchiveIndexStore:
    path: str

    def save(self, index: V2ArchiveIndex) -> None:
        destination = Path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                {
                    "documents": [asdict(document) for document in index.documents],
                    "feature_vocabulary": index.feature_vocabulary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> V2ArchiveIndex:
        source = Path(self.path)
        if not source.exists():
            return V2ArchiveIndex()
        payload = json.loads(source.read_text(encoding="utf-8"))
        return V2ArchiveIndex(
            documents=[V2IndexedDocument(**row) for row in payload.get("documents", [])],
            feature_vocabulary=payload.get("feature_vocabulary", {}),
        )
