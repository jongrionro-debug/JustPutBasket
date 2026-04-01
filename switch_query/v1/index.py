"""Local archive index implementations for the V1 scaffold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .models import V1ArchiveRecord


@dataclass(slots=True)
class InMemoryArchiveIndex:
    records: list[V1ArchiveRecord] = field(default_factory=list)

    def save(self, records: list[V1ArchiveRecord]) -> None:
        self.records = list(records)

    def load(self) -> list[V1ArchiveRecord]:
        return list(self.records)


@dataclass(slots=True)
class JsonArchiveIndexStore:
    path: str

    def save(self, records: list[V1ArchiveRecord]) -> None:
        destination = Path(self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps([asdict(record) for record in records], indent=2),
            encoding="utf-8",
        )

    def load(self) -> list[V1ArchiveRecord]:
        source = Path(self.path)
        if not source.exists():
            return []
        data = json.loads(source.read_text(encoding="utf-8"))
        return [V1ArchiveRecord(**record) for record in data]
