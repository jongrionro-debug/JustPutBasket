"""CLI helpers for building and running the V3 retrieval index."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re

from switch_query.v1.encoder import DEFAULT_SIGLIP2_MODEL, SigLIP2Encoder, SigLIP2EncoderConfig

from .index import build_archive_index as build_v3_archive_index
from .llm_parser import LuxiaV3QueryParser
from .models import (
    V3CandidateMode,
    V3CandidateTrace,
    V3PipelineInput,
    V3RankedResult,
)
from .pipeline import V3Pipeline, V3PipelineConfig
from .preprocessing import (
    backfill_archive_document_style_tags,
    read_archive_documents_jsonl,
    write_archive_documents_jsonl,
)
from .storage import JsonArchiveIndexStore


@dataclass(slots=True)
class BuildIndexResult:
    documents_path: str
    output_path: str
    model_name: str
    device: str
    batch_size: int
    document_count: int
    brand_count: int


@dataclass(slots=True)
class RunQueryResult:
    query_text: str
    index_path: str
    candidate_mode: V3CandidateMode
    output_path: str | None
    html_output_path: str | None
    candidate_output_path: str | None
    model_name: str
    device: str
    batch_size: int
    result_count: int
    candidate_count: int


@dataclass(slots=True)
class RunQuerySetResult:
    queries_path: str
    output_dir: str
    manifest_path: str
    candidate_judgments_path: str
    candidate_mode: V3CandidateMode
    model_name: str
    device: str
    batch_size: int
    query_count: int
    candidate_count: int


@dataclass(slots=True)
class CompareQueryModesResult:
    query_text: str
    index_path: str
    output_dir: str
    dashboard_path: str
    model_name: str
    device: str
    batch_size: int
    top_k: int
    mode_count: int


@dataclass(slots=True)
class BackfillStyleTagsResult:
    input_path: str
    output_path: str
    document_count: int
    updated_document_count: int
    updated_item_count: int


def build_archive_index(
    *,
    documents_path: str,
    output_path: str,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> BuildIndexResult:
    documents = read_archive_documents_jsonl(documents_path)
    if not documents:
        raise ValueError(f"No V3 archive documents found in {documents_path}")

    encoder = SigLIP2Encoder(
        SigLIP2EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )
    build_v3_archive_index(documents, encoder, JsonArchiveIndexStore(output_path))
    return BuildIndexResult(
        documents_path=str(Path(documents_path).resolve()),
        output_path=str(Path(output_path).resolve()),
        model_name=model_name,
        device=encoder.device,
        batch_size=batch_size,
        document_count=len(documents),
        brand_count=len({document.brand for document in documents}),
    )


def backfill_style_tags(
    *,
    input_path: str,
    output_path: str,
) -> BackfillStyleTagsResult:
    documents = read_archive_documents_jsonl(input_path)
    if not documents:
        raise ValueError(f"No V3 archive documents found in {input_path}")

    updated_documents = backfill_archive_document_style_tags(documents)
    write_archive_documents_jsonl(output_path, updated_documents)

    updated_document_count = 0
    updated_item_count = 0
    for before, after in zip(documents, updated_documents, strict=True):
        document_changed = False
        for before_item, after_item in zip(before.items, after.items, strict=True):
            if (
                before_item.style_tags != after_item.style_tags
                or before_item.style_concepts != after_item.style_concepts
                or before_item.color != after_item.color
            ):
                updated_item_count += 1
                document_changed = True
        if before.item_extraction_notes != after.item_extraction_notes and not document_changed:
            document_changed = True
        if document_changed:
            updated_document_count += 1

    return BackfillStyleTagsResult(
        input_path=str(Path(input_path).resolve()),
        output_path=str(Path(output_path).resolve()),
        document_count=len(documents),
        updated_document_count=updated_document_count,
        updated_item_count=updated_item_count,
    )


def run_query(
    *,
    index_path: str,
    query_text: str,
    stage: str,
    balance_score: float,
    user_uploaded_image: str | None,
    candidate_mode: V3CandidateMode,
    symbolic_candidate_k: int,
    embedding_candidate_k: int,
    output_path: str | None,
    html_output_path: str | None,
    candidate_output_path: str | None,
    model_name: str,
    device: str | None,
    batch_size: int,
    top_k: int,
) -> RunQueryResult:
    output = _execute_query(
        index_path=index_path,
        query_text=query_text,
        stage=stage,
        balance_score=balance_score,
        user_uploaded_image=user_uploaded_image,
        candidate_mode=candidate_mode,
        symbolic_candidate_k=symbolic_candidate_k,
        embedding_candidate_k=embedding_candidate_k,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        top_k=top_k,
    )
    if output_path:
        write_ranked_results_csv(output_path, output.top_results)
    if html_output_path:
        write_ranked_results_html(
            html_output_path,
            output.top_results,
            query_text=query_text,
            stage=stage,
            candidate_mode=candidate_mode,
        )
    if candidate_output_path:
        document_lookup = _load_document_lookup(index_path)
        write_candidate_traces_csv(
            candidate_output_path,
            output.candidate_traces,
            document_lookup=document_lookup,
            query_text=query_text,
            stage=stage,
            candidate_mode=candidate_mode,
        )
    return RunQueryResult(
        query_text=query_text,
        index_path=str(Path(index_path).resolve()),
        candidate_mode=candidate_mode,
        output_path=str(Path(output_path).resolve()) if output_path else None,
        html_output_path=str(Path(html_output_path).resolve()) if html_output_path else None,
        candidate_output_path=str(Path(candidate_output_path).resolve())
        if candidate_output_path
        else None,
        model_name=model_name,
        device=device or "",
        batch_size=batch_size,
        result_count=len(output.top_results),
        candidate_count=len(output.candidate_traces),
    )


def run_query_set(
    *,
    index_path: str,
    queries_path: str,
    output_dir: str,
    candidate_mode: V3CandidateMode,
    model_name: str,
    device: str | None,
    batch_size: int,
    top_k: int,
    symbolic_candidate_k: int,
    embedding_candidate_k: int,
    default_stage: str = "mood_board",
    default_balance_score: float = 0.0,
) -> RunQuerySetResult:
    query_rows = _read_query_rows(queries_path)
    if not query_rows:
        raise ValueError(f"No query rows found in {queries_path}")

    output_root = Path(output_dir)
    results_dir = output_root / "results"
    manifest_path = output_root / "query_manifest.csv"
    candidate_judgments_path = output_root / "candidate_judgments.csv"
    document_lookup = _load_document_lookup(index_path)

    manifest_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for query_row in query_rows:
        query_id = (query_row.get("query_id") or "").strip()
        query_text = (query_row.get("query_text") or "").strip()
        if not query_id:
            raise ValueError("Each query row must include query_id")
        if not query_text:
            raise ValueError(f"Query row '{query_id}' is missing query_text")

        stage = (query_row.get("stage") or default_stage).strip() or default_stage
        balance_score = _coerce_float(query_row.get("balance_score"), default=default_balance_score)
        file_stem = _safe_file_stem(query_id)
        csv_output_path = results_dir / f"{file_stem}.csv"
        html_output_path = results_dir / f"{file_stem}.html"

        output = _execute_query(
            index_path=index_path,
            query_text=query_text,
            stage=stage,
            balance_score=balance_score,
            user_uploaded_image=None,
            candidate_mode=candidate_mode,
            symbolic_candidate_k=symbolic_candidate_k,
            embedding_candidate_k=embedding_candidate_k,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            top_k=top_k,
        )
        write_ranked_results_csv(str(csv_output_path), output.top_results)
        write_ranked_results_html(
            str(html_output_path),
            output.top_results,
            query_text=query_text,
            stage=stage,
            candidate_mode=candidate_mode,
        )

        manifest_rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "stage": stage,
                "query_type": (query_row.get("query_type") or "").strip(),
                "expected_failure_type": (query_row.get("expected_failure_type") or "").strip(),
                "candidate_mode": candidate_mode,
                "csv_output_path": str(csv_output_path.resolve()),
                "html_output_path": str(html_output_path.resolve()),
                "result_count": len(output.top_results),
            }
        )

        for trace in output.candidate_traces:
            document = document_lookup.get(trace.image_id)
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "stage": stage,
                    "query_type": (query_row.get("query_type") or "").strip(),
                    "expected_failure_type": (query_row.get("expected_failure_type") or "").strip(),
                    "candidate_mode": candidate_mode,
                    "rank": trace.final_rank or "",
                    "image_id": trace.image_id,
                    "final_score": _format_optional_float(trace.final_score),
                    "brand": document.get("brand", ""),
                    "season_group": document.get("season_group", ""),
                    "file_path": document.get("file_path", ""),
                    "in_symbolic_pool": _serialize_bool(trace.in_symbolic_pool),
                    "symbolic_rank": trace.symbolic_rank or "",
                    "symbolic_score": _format_optional_float(trace.symbolic_score),
                    "in_embedding_pool": _serialize_bool(trace.in_embedding_pool),
                    "embedding_rank": trace.embedding_rank or "",
                    "embedding_score": _format_optional_float(trace.embedding_score),
                    "label": "",
                    "failure_type": "",
                    "notes": "",
                }
            )

    _write_dict_csv(
        manifest_path,
        fieldnames=[
            "query_id",
            "query_text",
            "stage",
            "query_type",
            "expected_failure_type",
            "candidate_mode",
            "csv_output_path",
            "html_output_path",
            "result_count",
        ],
        rows=manifest_rows,
    )
    _write_dict_csv(
        candidate_judgments_path,
        fieldnames=[
            "query_id",
            "query_text",
            "stage",
            "query_type",
            "expected_failure_type",
            "candidate_mode",
            "rank",
            "image_id",
            "final_score",
            "brand",
            "season_group",
            "file_path",
            "in_symbolic_pool",
            "symbolic_rank",
            "symbolic_score",
            "in_embedding_pool",
            "embedding_rank",
            "embedding_score",
            "label",
            "failure_type",
            "notes",
        ],
        rows=candidate_rows,
    )

    return RunQuerySetResult(
        queries_path=str(Path(queries_path).resolve()),
        output_dir=str(output_root.resolve()),
        manifest_path=str(manifest_path.resolve()),
        candidate_judgments_path=str(candidate_judgments_path.resolve()),
        candidate_mode=candidate_mode,
        model_name=model_name,
        device=device or "",
        batch_size=batch_size,
        query_count=len(query_rows),
        candidate_count=len(candidate_rows),
    )


def compare_query_modes(
    *,
    index_path: str,
    query_text: str,
    stage: str,
    balance_score: float,
    user_uploaded_image: str | None,
    output_dir: str,
    model_name: str,
    device: str | None,
    batch_size: int,
    top_k: int,
    symbolic_candidate_k: int,
    embedding_candidate_k: int,
) -> CompareQueryModesResult:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    mode_outputs: dict[str, object] = {}
    mode_artifacts: dict[str, dict[str, str]] = {}
    document_lookup = _load_document_lookup(index_path)

    for candidate_mode in ("symbolic_only", "embedding_only", "union"):
        mode_slug = str(candidate_mode)
        csv_output_path = output_root / f"{mode_slug}.csv"
        html_output_path = output_root / f"{mode_slug}.html"
        candidate_output_path = output_root / f"{mode_slug}_candidates.csv"
        output = _execute_query(
            index_path=index_path,
            query_text=query_text,
            stage=stage,
            balance_score=balance_score,
            user_uploaded_image=user_uploaded_image,
            candidate_mode=mode_slug,  # type: ignore[arg-type]
            symbolic_candidate_k=symbolic_candidate_k,
            embedding_candidate_k=embedding_candidate_k,
            model_name=model_name,
            device=device,
            batch_size=batch_size,
            top_k=top_k,
        )
        write_ranked_results_csv(str(csv_output_path), output.top_results)
        write_ranked_results_html(
            str(html_output_path),
            output.top_results,
            query_text=query_text,
            stage=stage,
            candidate_mode=mode_slug,  # type: ignore[arg-type]
        )
        write_candidate_traces_csv(
            str(candidate_output_path),
            output.candidate_traces,
            document_lookup=document_lookup,
            query_text=query_text,
            stage=stage,
            candidate_mode=mode_slug,  # type: ignore[arg-type]
        )
        mode_outputs[mode_slug] = output
        mode_artifacts[mode_slug] = {
            "csv_output_path": str(csv_output_path.resolve()),
            "html_output_path": str(html_output_path.resolve()),
            "candidate_output_path": str(candidate_output_path.resolve()),
        }

    dashboard_path = output_root / "compare_modes.html"
    write_mode_comparison_html(
        str(dashboard_path),
        mode_outputs=mode_outputs,
        mode_artifacts=mode_artifacts,
        query_text=query_text,
        stage=stage,
    )
    return CompareQueryModesResult(
        query_text=query_text,
        index_path=str(Path(index_path).resolve()),
        output_dir=str(output_root.resolve()),
        dashboard_path=str(dashboard_path.resolve()),
        model_name=model_name,
        device=device or "",
        batch_size=batch_size,
        top_k=top_k,
        mode_count=3,
    )


def write_ranked_results_csv(path: str, results: list[V3RankedResult]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "image_id",
                "score",
                "brand",
                "season_group",
                "file_path",
                "item_assignments",
                "score_breakdown",
                "match_reasons",
                "explanation",
            ],
        )
        writer.writeheader()
        for index, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": index,
                    "image_id": result.image_id,
                    "score": result.score,
                    "brand": result.brand,
                    "season_group": result.season_group,
                    "file_path": result.file_path,
                    "item_assignments": _serialize_item_assignments(result),
                    "score_breakdown": _serialize_score_breakdown(result.score_breakdown),
                    "match_reasons": "|".join(result.match_reasons),
                    "explanation": result.explanation,
                }
            )


def write_ranked_results_html(
    path: str,
    results: list[V3RankedResult],
    *,
    query_text: str,
    stage: str,
    candidate_mode: V3CandidateMode,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for index, result in enumerate(results, start=1):
        image_uri = Path(result.file_path).resolve().as_uri()
        cards.append(
            f"""
            <article class="card">
              <img src="{escape(image_uri)}" alt="{escape(result.image_id)}" loading="lazy">
              <div class="meta">
                <div><strong>rank</strong> {index}</div>
                <div><strong>image_id</strong> {escape(result.image_id)}</div>
                <div><strong>brand</strong> {escape(result.brand)}</div>
                <div><strong>score</strong> {result.score:.6f}</div>
                <div><strong>season_group</strong> {escape(result.season_group)}</div>
                <div><strong>item_assignments</strong> {escape(_serialize_item_assignments(result))}</div>
                <div class="explanation">{escape(result.explanation)}</div>
                <div><strong>score_breakdown</strong> {escape(_serialize_score_breakdown(result.score_breakdown))}</div>
                <div><strong>match_reasons</strong> {escape(" | ".join(result.match_reasons) or "none")}</div>
                <div class="path">{escape(result.file_path)}</div>
              </div>
            </article>
            """.strip()
        )

    document = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>V3 Query Preview</title>
        <style>
          :root {{
            color-scheme: light;
            --bg: #f4f1ec;
            --surface: rgba(255, 255, 255, 0.92);
            --text: #201a17;
            --muted: #6b635e;
            --line: rgba(32, 26, 23, 0.12);
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            background:
              radial-gradient(circle at top left, rgba(195, 167, 125, 0.18), transparent 28%),
              linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
            color: var(--text);
          }}
          main {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 32px 20px 60px;
          }}
          header {{
            margin-bottom: 28px;
          }}
          h1 {{
            margin: 0 0 8px;
            font-size: 32px;
            line-height: 1.1;
          }}
          .summary {{
            color: var(--muted);
            font-size: 14px;
          }}
          .query {{
            margin-top: 10px;
            font-size: 18px;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 18px;
          }}
          .card {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(32, 26, 23, 0.08);
          }}
          img {{
            display: block;
            width: 100%;
            aspect-ratio: 3 / 4;
            object-fit: cover;
            background: #e8dfd2;
          }}
          .meta {{
            padding: 14px;
            display: grid;
            gap: 6px;
            font-size: 13px;
          }}
          .explanation {{
            color: var(--muted);
            line-height: 1.45;
          }}
          .path {{
            color: var(--muted);
            font-size: 11px;
            word-break: break-all;
            margin-top: 2px;
          }}
        </style>
      </head>
      <body>
        <main>
          <header>
            <h1>V3 Retrieval Preview</h1>
            <div class="summary">stage: {escape(stage)} | candidate_mode: {escape(candidate_mode)} | results: {len(results)}</div>
            <div class="query">{escape(query_text)}</div>
          </header>
          <section class="grid">
            {' '.join(cards)}
          </section>
        </main>
      </body>
    </html>
    """.strip()
    destination.write_text(document, encoding="utf-8")


def write_candidate_traces_csv(
    path: str,
    candidate_traces: list[V3CandidateTrace],
    *,
    document_lookup: dict[str, dict[str, str]],
    query_text: str,
    stage: str,
    candidate_mode: V3CandidateMode,
) -> None:
    rows = []
    for trace in candidate_traces:
        document = document_lookup.get(trace.image_id, {})
        rows.append(
            {
                "query_text": query_text,
                "stage": stage,
                "candidate_mode": candidate_mode,
                "rank": trace.final_rank or "",
                "image_id": trace.image_id,
                "final_score": _format_optional_float(trace.final_score),
                "brand": document.get("brand", ""),
                "season_group": document.get("season_group", ""),
                "file_path": document.get("file_path", ""),
                "in_symbolic_pool": _serialize_bool(trace.in_symbolic_pool),
                "symbolic_rank": trace.symbolic_rank or "",
                "symbolic_score": _format_optional_float(trace.symbolic_score),
                "in_embedding_pool": _serialize_bool(trace.in_embedding_pool),
                "embedding_rank": trace.embedding_rank or "",
                "embedding_score": _format_optional_float(trace.embedding_score),
            }
        )
    _write_dict_csv(
        Path(path),
        fieldnames=[
            "query_text",
            "stage",
            "candidate_mode",
            "rank",
            "image_id",
            "final_score",
            "brand",
            "season_group",
            "file_path",
            "in_symbolic_pool",
            "symbolic_rank",
            "symbolic_score",
            "in_embedding_pool",
            "embedding_rank",
            "embedding_score",
        ],
        rows=rows,
    )


def write_mode_comparison_html(
    path: str,
    *,
    mode_outputs: dict[str, object],
    mode_artifacts: dict[str, dict[str, str]],
    query_text: str,
    stage: str,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    mode_sections = []
    for candidate_mode in ("symbolic_only", "embedding_only", "union"):
        output = mode_outputs[candidate_mode]
        artifacts = mode_artifacts[candidate_mode]
        top_results = getattr(output, "top_results")
        candidate_traces = getattr(output, "candidate_traces")
        retrieval_metadata = getattr(output, "retrieval_metadata")
        cards = []
        for rank, result in enumerate(top_results, start=1):
            image_uri = Path(result.file_path).resolve().as_uri()
            cards.append(
                f"""
                <article class="card">
                  <img src="{escape(image_uri)}" alt="{escape(result.image_id)}" loading="lazy">
                  <div class="meta">
                    <div><strong>rank</strong> {rank}</div>
                    <div><strong>image_id</strong> {escape(result.image_id)}</div>
                    <div><strong>brand</strong> {escape(result.brand)}</div>
                    <div><strong>score</strong> {result.score:.6f}</div>
                    <div><strong>item_assignments</strong> {escape(_serialize_item_assignments(result))}</div>
                    <div class="explanation">{escape(result.explanation)}</div>
                  </div>
                </article>
                """.strip()
            )
        mode_sections.append(
            f"""
            <section class="mode-panel" id="{escape(candidate_mode)}">
              <div class="mode-header">
                <div>
                  <h2>{escape(candidate_mode)}</h2>
                  <div class="summary">
                    final_results={len(top_results)} |
                    symbolic_candidates={retrieval_metadata.get("symbolic_candidate_count", 0)} |
                    embedding_candidates={retrieval_metadata.get("embedding_candidate_count", 0)} |
                    union_candidates={retrieval_metadata.get("union_candidate_count", 0)}
                  </div>
                </div>
                <div class="links">
                  <a href="{escape(Path(artifacts['html_output_path']).resolve().as_uri())}">mode html</a>
                  <a href="{escape(Path(artifacts['csv_output_path']).resolve().as_uri())}">results csv</a>
                  <a href="{escape(Path(artifacts['candidate_output_path']).resolve().as_uri())}">candidates csv</a>
                </div>
              </div>
              <details class="candidate-details">
                <summary>candidate traces ({len(candidate_traces)})</summary>
                <div class="trace-table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>image_id</th>
                        <th>final_rank</th>
                        <th>final_score</th>
                        <th>symbolic</th>
                        <th>symbolic_rank</th>
                        <th>symbolic_score</th>
                        <th>embedding</th>
                        <th>embedding_rank</th>
                        <th>embedding_score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {"".join(
                          f"<tr>"
                          f"<td>{escape(trace.image_id)}</td>"
                          f"<td>{trace.final_rank or ''}</td>"
                          f"<td>{escape(_format_optional_float(trace.final_score))}</td>"
                          f"<td>{_serialize_bool(trace.in_symbolic_pool)}</td>"
                          f"<td>{trace.symbolic_rank or ''}</td>"
                          f"<td>{escape(_format_optional_float(trace.symbolic_score))}</td>"
                          f"<td>{_serialize_bool(trace.in_embedding_pool)}</td>"
                          f"<td>{trace.embedding_rank or ''}</td>"
                          f"<td>{escape(_format_optional_float(trace.embedding_score))}</td>"
                          f"</tr>"
                          for trace in candidate_traces
                      )}
                    </tbody>
                  </table>
                </div>
              </details>
              <div class="grid">
                {' '.join(cards) if cards else '<div class="empty">no results</div>'}
              </div>
            </section>
            """.strip()
        )

    document = f"""
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>V3 Mode Comparison</title>
        <style>
          :root {{
            color-scheme: light;
            --bg: #f4f1ec;
            --surface: rgba(255, 255, 255, 0.94);
            --surface-strong: #ffffff;
            --text: #201a17;
            --muted: #6b635e;
            --line: rgba(32, 26, 23, 0.12);
            --accent: #8f5f36;
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            background:
              radial-gradient(circle at top left, rgba(195, 167, 125, 0.18), transparent 28%),
              linear-gradient(180deg, #faf6ef 0%, var(--bg) 100%);
            color: var(--text);
          }}
          main {{
            max-width: 1680px;
            margin: 0 auto;
            padding: 28px 18px 60px;
          }}
          h1 {{
            margin: 0 0 8px;
            font-size: 34px;
          }}
          .summary {{
            color: var(--muted);
            font-size: 14px;
          }}
          .query {{
            margin-top: 10px;
            font-size: 20px;
          }}
          .mode-nav {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin: 24px 0;
          }}
          .mode-nav a {{
            color: var(--accent);
            text-decoration: none;
            background: rgba(143, 95, 54, 0.08);
            border: 1px solid rgba(143, 95, 54, 0.18);
            padding: 10px 14px;
            border-radius: 999px;
            font-size: 14px;
          }}
          .mode-panel {{
            margin-top: 22px;
            padding: 18px;
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(32, 26, 23, 0.06);
          }}
          .mode-header {{
            display: flex;
            gap: 16px;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 14px;
          }}
          .mode-header h2 {{
            margin: 0 0 6px;
            font-size: 24px;
          }}
          .links {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
          }}
          .links a {{
            color: var(--accent);
            text-decoration: none;
            font-size: 13px;
          }}
          .candidate-details {{
            margin: 10px 0 16px;
          }}
          .candidate-details summary {{
            cursor: pointer;
            color: var(--muted);
          }}
          .trace-table-wrap {{
            overflow-x: auto;
            margin-top: 10px;
            background: var(--surface-strong);
            border: 1px solid var(--line);
            border-radius: 14px;
          }}
          table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
          }}
          th, td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--line);
            text-align: left;
            white-space: nowrap;
          }}
          th {{
            background: #f7f2ea;
          }}
          .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
            gap: 16px;
          }}
          .card {{
            background: var(--surface-strong);
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
          }}
          img {{
            display: block;
            width: 100%;
            aspect-ratio: 3 / 4;
            object-fit: cover;
            background: #e8dfd2;
          }}
          .meta {{
            padding: 12px;
            display: grid;
            gap: 6px;
            font-size: 12px;
          }}
          .explanation {{
            color: var(--muted);
            line-height: 1.45;
          }}
          .empty {{
            color: var(--muted);
            padding: 14px;
          }}
        </style>
      </head>
      <body>
        <main>
          <header>
            <h1>V3 Mode Comparison</h1>
            <div class="summary">stage: {escape(stage)} | modes: 3</div>
            <div class="query">{escape(query_text)}</div>
          </header>
          <nav class="mode-nav">
            <a href="#symbolic_only">symbolic_only</a>
            <a href="#embedding_only">embedding_only</a>
            <a href="#union">union</a>
          </nav>
          {' '.join(mode_sections)}
        </main>
      </body>
    </html>
    """.strip()
    destination.write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 item-aware retrieval helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Build a V3 archive image index from item-enriched documents JSONL.",
    )
    build_index_parser.add_argument("--documents", required=True)
    build_index_parser.add_argument("--output", required=True)
    build_index_parser.add_argument("--model-name", default=DEFAULT_SIGLIP2_MODEL)
    build_index_parser.add_argument("--device")
    build_index_parser.add_argument("--batch-size", type=int, default=8)

    backfill_style_tags_parser = subparsers.add_parser(
        "backfill-style-tags",
        help="Backfill item style_tags in item-enriched documents JSONL without re-running VLM extraction.",
    )
    backfill_style_tags_parser.add_argument("--input", required=True)
    backfill_style_tags_parser.add_argument("--output", required=True)

    run_query_parser = subparsers.add_parser(
        "run-query",
        help="Run a query against a saved V3 archive index.",
    )
    run_query_parser.add_argument("--index", required=True)
    run_query_parser.add_argument("--query", required=True)
    run_query_parser.add_argument(
        "--stage",
        choices=["mood_board", "sketch_stage"],
        default="mood_board",
    )
    run_query_parser.add_argument("--balance-score", type=float, default=0.0)
    run_query_parser.add_argument("--user-uploaded-image")
    run_query_parser.add_argument(
        "--candidate-mode",
        choices=["symbolic_only", "embedding_only", "union"],
        default="union",
    )
    run_query_parser.add_argument("--symbolic-candidate-k", type=int, default=100)
    run_query_parser.add_argument("--embedding-candidate-k", type=int, default=100)
    run_query_parser.add_argument("--output")
    run_query_parser.add_argument("--html-output")
    run_query_parser.add_argument("--candidate-output")
    run_query_parser.add_argument("--model-name", default=DEFAULT_SIGLIP2_MODEL)
    run_query_parser.add_argument("--device")
    run_query_parser.add_argument("--batch-size", type=int, default=8)
    run_query_parser.add_argument("--top-k", type=int, default=20)

    run_query_set_parser = subparsers.add_parser(
        "run-query-set",
        help="Run a CSV of queries and emit per-query previews plus a labeling sheet.",
    )
    run_query_set_parser.add_argument("--index", required=True)
    run_query_set_parser.add_argument("--queries", required=True)
    run_query_set_parser.add_argument("--output-dir", required=True)
    run_query_set_parser.add_argument(
        "--candidate-mode",
        choices=["symbolic_only", "embedding_only", "union"],
        default="union",
    )
    run_query_set_parser.add_argument("--model-name", default=DEFAULT_SIGLIP2_MODEL)
    run_query_set_parser.add_argument("--device")
    run_query_set_parser.add_argument("--batch-size", type=int, default=8)
    run_query_set_parser.add_argument("--top-k", type=int, default=20)
    run_query_set_parser.add_argument("--symbolic-candidate-k", type=int, default=100)
    run_query_set_parser.add_argument("--embedding-candidate-k", type=int, default=100)
    run_query_set_parser.add_argument(
        "--default-stage",
        choices=["mood_board", "sketch_stage"],
        default="mood_board",
    )
    run_query_set_parser.add_argument("--default-balance-score", type=float, default=0.0)

    compare_query_parser = subparsers.add_parser(
        "compare-query",
        help="Run all V3 candidate modes and build a comparison dashboard HTML.",
    )
    compare_query_parser.add_argument("--index", required=True)
    compare_query_parser.add_argument("--query", required=True)
    compare_query_parser.add_argument(
        "--stage",
        choices=["mood_board", "sketch_stage"],
        default="mood_board",
    )
    compare_query_parser.add_argument("--balance-score", type=float, default=0.0)
    compare_query_parser.add_argument("--user-uploaded-image")
    compare_query_parser.add_argument("--output-dir", required=True)
    compare_query_parser.add_argument("--model-name", default=DEFAULT_SIGLIP2_MODEL)
    compare_query_parser.add_argument("--device")
    compare_query_parser.add_argument("--batch-size", type=int, default=8)
    compare_query_parser.add_argument("--top-k", type=int, default=20)
    compare_query_parser.add_argument("--symbolic-candidate-k", type=int, default=100)
    compare_query_parser.add_argument("--embedding-candidate-k", type=int, default=100)

    args = parser.parse_args()

    if args.command == "build-index":
        result = build_archive_index(
            documents_path=args.documents,
            output_path=args.output,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(
            "\n".join(
                [
                    f"documents_path={result.documents_path}",
                    f"output_path={result.output_path}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"document_count={result.document_count}",
                    f"brand_count={result.brand_count}",
                ]
            )
        )
        return

    if args.command == "backfill-style-tags":
        result = backfill_style_tags(
            input_path=args.input,
            output_path=args.output,
        )
        print(
            "\n".join(
                [
                    f"input_path={result.input_path}",
                    f"output_path={result.output_path}",
                    f"document_count={result.document_count}",
                    f"updated_document_count={result.updated_document_count}",
                    f"updated_item_count={result.updated_item_count}",
                ]
            )
        )
        return

    if args.command == "run-query":
        result = run_query(
            index_path=args.index,
            query_text=args.query,
            stage=args.stage,
            balance_score=args.balance_score,
            user_uploaded_image=args.user_uploaded_image,
            candidate_mode=args.candidate_mode,
            symbolic_candidate_k=args.symbolic_candidate_k,
            embedding_candidate_k=args.embedding_candidate_k,
            output_path=args.output,
            html_output_path=args.html_output,
            candidate_output_path=args.candidate_output,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
        print(
            "\n".join(
                [
                    f"query_text={result.query_text}",
                    f"index_path={result.index_path}",
                    f"candidate_mode={result.candidate_mode}",
                    f"output_path={result.output_path or ''}",
                    f"html_output_path={result.html_output_path or ''}",
                    f"candidate_output_path={result.candidate_output_path or ''}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"result_count={result.result_count}",
                    f"candidate_count={result.candidate_count}",
                ]
            )
        )
        return

    if args.command == "run-query-set":
        result = run_query_set(
            index_path=args.index,
            queries_path=args.queries,
            output_dir=args.output_dir,
            candidate_mode=args.candidate_mode,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            symbolic_candidate_k=args.symbolic_candidate_k,
            embedding_candidate_k=args.embedding_candidate_k,
            default_stage=args.default_stage,
            default_balance_score=args.default_balance_score,
        )
        print(
            "\n".join(
                [
                    f"queries_path={result.queries_path}",
                    f"output_dir={result.output_dir}",
                    f"manifest_path={result.manifest_path}",
                    f"candidate_judgments_path={result.candidate_judgments_path}",
                    f"candidate_mode={result.candidate_mode}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"query_count={result.query_count}",
                    f"candidate_count={result.candidate_count}",
                ]
            )
        )
        return

    if args.command == "compare-query":
        result = compare_query_modes(
            index_path=args.index,
            query_text=args.query,
            stage=args.stage,
            balance_score=args.balance_score,
            user_uploaded_image=args.user_uploaded_image,
            output_dir=args.output_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            symbolic_candidate_k=args.symbolic_candidate_k,
            embedding_candidate_k=args.embedding_candidate_k,
        )
        print(
            "\n".join(
                [
                    f"query_text={result.query_text}",
                    f"index_path={result.index_path}",
                    f"output_dir={result.output_dir}",
                    f"dashboard_path={result.dashboard_path}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"top_k={result.top_k}",
                    f"mode_count={result.mode_count}",
                ]
            )
        )


def _execute_query(
    *,
    index_path: str,
    query_text: str,
    stage: str,
    balance_score: float,
    user_uploaded_image: str | None,
    candidate_mode: V3CandidateMode,
    symbolic_candidate_k: int,
    embedding_candidate_k: int,
    model_name: str,
    device: str | None,
    batch_size: int,
    top_k: int,
):
    encoder = SigLIP2Encoder(
        SigLIP2EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )
    pipeline = V3Pipeline(
        index_store=JsonArchiveIndexStore(index_path),
        encoder=encoder,
        parser=LuxiaV3QueryParser(),
        config=V3PipelineConfig(
            top_k=top_k,
            candidate_mode=candidate_mode,
            symbolic_candidate_k=symbolic_candidate_k,
            embedding_candidate_k=embedding_candidate_k,
        ),
    )
    return pipeline.run(
        V3PipelineInput(
            query_text=query_text,
            stage=stage,  # type: ignore[arg-type]
            balance_score=balance_score,
            user_uploaded_image=user_uploaded_image,
        )
    )


def _load_document_lookup(index_path: str) -> dict[str, dict[str, str]]:
    index = JsonArchiveIndexStore(index_path).load()
    return {
        document.image_id: {
            "brand": document.brand,
            "season_group": document.season_group,
            "file_path": document.file_path,
        }
        for document in index.documents
    }


def _read_query_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _coerce_float(raw_value: str | None, *, default: float) -> float:
    if raw_value is None or not raw_value.strip():
        return default
    return float(raw_value)


def _safe_file_stem(raw_value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value.strip())
    return collapsed.strip("._") or "query"


def _write_dict_csv(
    path: Path,
    *,
    fieldnames: list[str],
    rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _serialize_item_assignments(result: V3RankedResult) -> str:
    payload = [
        {
            "target_item_id": assignment.target_item_id,
            "status": assignment.status,
            "matched_item_id": assignment.matched_item_id,
            "matched_category": assignment.matched_category,
        }
        for assignment in result.item_assignments
    ]
    return json.dumps(payload, ensure_ascii=False)


def _serialize_score_breakdown(payload: dict[str, float]) -> str:
    return "|".join(f"{key}={value:+.1f}" for key, value in payload.items())


def _serialize_bool(value: bool) -> str:
    return "true" if value else "false"


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    main()
