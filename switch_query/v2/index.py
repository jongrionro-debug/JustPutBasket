"""Index builders for the V2 text/tag retrieval pipeline."""

from __future__ import annotations

from typing import Sequence

from .documents import build_feature_vocabulary
from .models import TextEncoder, V2ArchiveDocument, V2ArchiveIndex, V2IndexedDocument


def build_archive_index(
    documents: Sequence[V2ArchiveDocument],
    encoder: TextEncoder | None,
    index_store,
) -> V2ArchiveIndex:
    if encoder is None:
        vectors = [[] for _ in documents]
    else:
        document_texts = [document.document_text for document in documents]
        vectors = encoder.encode_text(document_texts)
        if len(vectors) != len(documents):
            raise RuntimeError("Text encoder output size did not match archive document count")

    indexed_documents = [
        V2IndexedDocument(
            image_id=document.image_id,
            file_path=document.file_path,
            brand=document.brand,
            season_group=document.season_group,
            canonical_tags=dict(document.canonical_tags),
            raw_tags=dict(document.raw_tags),
            document_text=document.document_text,
            vector=vector,
        )
        for document, vector in zip(documents, vectors, strict=True)
    ]
    index = V2ArchiveIndex(
        documents=indexed_documents,
        feature_vocabulary=build_feature_vocabulary(documents=documents),
    )
    index_store.save(index)
    return index
