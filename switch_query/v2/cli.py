"""CLI helpers for building and running the V2 text/tag retrieval index."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path
import re

from switch_query.tagging import NormalizedTagRow

from .documents import build_archive_documents, write_archive_documents
from .encoder import SigLIP2TextEncoder, SigLIP2TextEncoderConfig
from .index import build_archive_index as build_v2_archive_index
from .llm_parser import LuxiaQueryParser
from .models import V2PipelineInput, V2RankedResult
from .pipeline import V2Pipeline, V2PipelineConfig
from .storage import JsonArchiveIndexStore


@dataclass(slots=True)
class BuildIndexResult:
    normalized_tags_path: str
    output_path: str
    documents_output_path: str | None
    embeddings_enabled: bool
    model_name: str
    device: str
    batch_size: int
    document_count: int
    brand_count: int


@dataclass(slots=True)
class RunQueryResult:
    query_text: str
    index_path: str
    output_path: str | None
    html_output_path: str | None
    parser_provider: str
    model_name: str
    device: str
    batch_size: int
    result_count: int


@dataclass(slots=True)
class RunQuerySetResult:
    queries_path: str
    output_dir: str
    manifest_path: str
    candidate_judgments_path: str
    parser_provider: str
    model_name: str
    device: str
    batch_size: int
    query_count: int
    candidate_count: int


def build_archive_index(
    *,
    normalized_tags_path: str,
    output_path: str,
    documents_output_path: str | None,
    model_name: str | None,
    device: str | None,
    batch_size: int,
    use_embeddings: bool = False,
) -> BuildIndexResult:
    rows = _read_normalized_rows(normalized_tags_path)
    if not rows:
        raise ValueError(f"No normalized rows found in {normalized_tags_path}")

    documents = build_archive_documents(rows)
    if documents_output_path:
        write_archive_documents(documents_output_path, documents)

    encoder = None
    resolved_model_name = ""
    resolved_device = device or ""
    if use_embeddings:
        encoder = SigLIP2TextEncoder(
            SigLIP2TextEncoderConfig(
                model_name=model_name or "google/siglip2-base-patch16-224",
                device=device,
                batch_size=batch_size,
            )
        )
        resolved_model_name = model_name or "google/siglip2-base-patch16-224"
        resolved_device = encoder.device
    build_v2_archive_index(documents, encoder, JsonArchiveIndexStore(output_path))
    return BuildIndexResult(
        normalized_tags_path=str(Path(normalized_tags_path).resolve()),
        output_path=str(Path(output_path).resolve()),
        documents_output_path=str(Path(documents_output_path).resolve())
        if documents_output_path
        else None,
        embeddings_enabled=use_embeddings,
        model_name=resolved_model_name,
        device=resolved_device,
        batch_size=batch_size,
        document_count=len(documents),
        brand_count=len({row.brand for row in rows}),
    )


def run_query(
    *,
    index_path: str,
    query_text: str,
    stage: str,
    balance_score: float,
    user_uploaded_image: str | None,
    output_path: str | None,
    html_output_path: str | None,
    parser_provider: str = "luxia",
    model_name: str | None,
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
        parser_provider=parser_provider,
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
        )
    return RunQueryResult(
        query_text=query_text,
        index_path=str(Path(index_path).resolve()),
        output_path=str(Path(output_path).resolve()) if output_path else None,
        html_output_path=str(Path(html_output_path).resolve()) if html_output_path else None,
        parser_provider=parser_provider,
        model_name=model_name or "",
        device=device or "",
        batch_size=batch_size,
        result_count=len(output.top_results),
    )


def run_query_set(
    *,
    index_path: str,
    queries_path: str,
    output_dir: str,
    parser_provider: str = "luxia",
    model_name: str | None,
    device: str | None,
    batch_size: int,
    top_k: int,
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

    manifest_rows: list[dict[str, str | int]] = []
    candidate_rows: list[dict[str, str | int | float]] = []

    for query_row in query_rows:
        query_id = (query_row.get("query_id") or "").strip()
        query_text = (query_row.get("query_text") or "").strip()
        if not query_id:
            raise ValueError("Each query row must include query_id")
        if not query_text:
            raise ValueError(f"Query row '{query_id}' is missing query_text")

        stage = (query_row.get("stage") or default_stage).strip() or default_stage
        balance_score = _coerce_float(
            query_row.get("balance_score"),
            default=default_balance_score,
        )
        file_stem = _safe_file_stem(query_id)
        csv_output_path = results_dir / f"{file_stem}.csv"
        html_output_path = results_dir / f"{file_stem}.html"

        output = _execute_query(
            index_path=index_path,
            query_text=query_text,
            stage=stage,
            balance_score=balance_score,
            user_uploaded_image=None,
            parser_provider=parser_provider,
            top_k=top_k,
        )
        write_ranked_results_csv(str(csv_output_path), output.top_results)
        write_ranked_results_html(
            str(html_output_path),
            output.top_results,
            query_text=query_text,
            stage=stage,
        )

        manifest_rows.append(
            {
                "query_id": query_id,
                "query_text": query_text,
                "stage": stage,
                "query_type": (query_row.get("query_type") or "").strip(),
                "expected_failure_type": (query_row.get("expected_failure_type") or "").strip(),
                "csv_output_path": str(csv_output_path.resolve()),
                "html_output_path": str(html_output_path.resolve()),
                "result_count": len(output.top_results),
            }
        )

        for rank, result in enumerate(output.top_results, start=1):
            candidate_rows.append(
                {
                    "query_id": query_id,
                    "query_text": query_text,
                    "stage": stage,
                    "query_type": (query_row.get("query_type") or "").strip(),
                    "expected_failure_type": (query_row.get("expected_failure_type") or "").strip(),
                    "rank": rank,
                    "image_id": result.image_id,
                    "score": result.score,
                    "brand": result.brand,
                    "season_group": result.season_group,
                    "file_path": result.file_path,
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
            "rank",
            "image_id",
            "score",
            "brand",
            "season_group",
            "file_path",
            "label",
            "failure_type",
            "notes",
        ],
        rows=candidate_rows,
    )

    if parser_provider != "luxia":
        raise ValueError(f"Unsupported parser provider: {parser_provider}")
    return RunQuerySetResult(
        queries_path=str(Path(queries_path).resolve()),
        output_dir=str(output_root.resolve()),
        manifest_path=str(manifest_path.resolve()),
        candidate_judgments_path=str(candidate_judgments_path.resolve()),
        parser_provider=parser_provider,
        model_name=model_name or "",
        device=device or "",
        batch_size=batch_size,
        query_count=len(query_rows),
        candidate_count=len(candidate_rows),
    )


def write_ranked_results_csv(path: str, results: list[V2RankedResult]) -> None:
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
                "matched_attributes",
                "mismatched_attributes",
                "missing_attributes",
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
                    "matched_attributes": _serialize_dict(result.matched_attributes),
                    "mismatched_attributes": _serialize_dict(result.mismatched_attributes),
                    "missing_attributes": _serialize_dict(result.missing_attributes),
                    "score_breakdown": _serialize_score_breakdown(result.score_breakdown),
                    "match_reasons": "|".join(result.match_reasons),
                    "explanation": result.explanation,
                }
            )


def write_ranked_results_html(
    path: str,
    results: list[V2RankedResult],
    *,
    query_text: str,
    stage: str,
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
        <title>V2 Query Preview</title>
        <style>
          :root {{
            color-scheme: light;
            --bg: #f5f1ea;
            --surface: rgba(255, 255, 255, 0.9);
            --text: #1f1a17;
            --muted: #6f655e;
            --line: rgba(31, 26, 23, 0.12);
          }}
          * {{ box-sizing: border-box; }}
          body {{
            margin: 0;
            font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
            background:
              radial-gradient(circle at top left, rgba(196, 167, 125, 0.22), transparent 28%),
              linear-gradient(180deg, #f9f5ee 0%, var(--bg) 100%);
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
            box-shadow: 0 10px 30px rgba(31, 26, 23, 0.08);
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
            <h1>V2 Retrieval Preview</h1>
            <div class="summary">stage: {escape(stage)} | results: {len(results)}</div>
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


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 text/tag retrieval helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Build a V2 archive text index from normalized tag rows.",
    )
    build_index_parser.add_argument("--normalized-tags", required=True)
    build_index_parser.add_argument("--output", required=True)
    build_index_parser.add_argument("--documents-output")
    build_index_parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Optional: build document vectors for future rerank experiments.",
    )
    build_index_parser.add_argument(
        "--model-name",
        default=None,
        help="Optional embedding model name. Ignored unless --use-embeddings is set.",
    )
    build_index_parser.add_argument("--device")
    build_index_parser.add_argument("--batch-size", type=int, default=8)

    run_query_parser = subparsers.add_parser(
        "run-query",
        help="Run a query against a saved V2 archive index.",
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
    run_query_parser.add_argument("--output")
    run_query_parser.add_argument("--html-output")
    run_query_parser.add_argument(
        "--parser-provider",
        choices=["luxia"],
        default="luxia",
    )
    run_query_parser.add_argument(
        "--model-name",
        default=None,
        help="Deprecated for query runtime. Query ranking no longer uses embeddings in MVP.",
    )
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
        "--parser-provider",
        choices=["luxia"],
        default="luxia",
    )
    run_query_set_parser.add_argument(
        "--model-name",
        default=None,
        help="Deprecated for query runtime. Query ranking no longer uses embeddings in MVP.",
    )
    run_query_set_parser.add_argument("--device")
    run_query_set_parser.add_argument("--batch-size", type=int, default=8)
    run_query_set_parser.add_argument("--top-k", type=int, default=20)
    run_query_set_parser.add_argument(
        "--default-stage",
        choices=["mood_board", "sketch_stage"],
        default="mood_board",
    )
    run_query_set_parser.add_argument("--default-balance-score", type=float, default=0.0)

    args = parser.parse_args()

    if args.command == "build-index":
        result = build_archive_index(
            normalized_tags_path=args.normalized_tags,
            output_path=args.output,
            documents_output_path=args.documents_output,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            use_embeddings=args.use_embeddings,
        )
        print(
            "\n".join(
                [
                    f"normalized_tags_path={result.normalized_tags_path}",
                    f"output_path={result.output_path}",
                    f"documents_output_path={result.documents_output_path or ''}",
                    f"embeddings_enabled={result.embeddings_enabled}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"document_count={result.document_count}",
                    f"brand_count={result.brand_count}",
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
            output_path=args.output,
            html_output_path=args.html_output,
            parser_provider=args.parser_provider,
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
                    f"output_path={result.output_path or ''}",
                    f"html_output_path={result.html_output_path or ''}",
                    f"parser_provider={result.parser_provider}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"result_count={result.result_count}",
                ]
            )
        )
        return

    if args.command == "run-query-set":
        result = run_query_set(
            index_path=args.index,
            queries_path=args.queries,
            output_dir=args.output_dir,
            parser_provider=args.parser_provider,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
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
                    f"parser_provider={result.parser_provider}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"query_count={result.query_count}",
                    f"candidate_count={result.candidate_count}",
                ]
            )
        )


def _read_normalized_rows(path: str) -> list[NormalizedTagRow]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [NormalizedTagRow(**row) for row in reader]


def _read_query_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _execute_query(
    *,
    index_path: str,
    query_text: str,
    stage: str,
    balance_score: float,
    user_uploaded_image: str | None,
    parser_provider: str,
    top_k: int,
):
    if parser_provider != "luxia":
        raise ValueError(f"Unsupported parser provider: {parser_provider}")
    pipeline = V2Pipeline(
        index_store=JsonArchiveIndexStore(index_path),
        parser=LuxiaQueryParser() if parser_provider == "luxia" else None,
        config=V2PipelineConfig(top_k=top_k),
    )
    return pipeline.run(
        V2PipelineInput(
            query_text=query_text,
            stage=stage,  # type: ignore[arg-type]
            balance_score=balance_score,
            user_uploaded_image=user_uploaded_image,
        )
    )


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


def _serialize_dict(payload: dict[str, str]) -> str:
    return "|".join(f"{key}={value}" for key, value in payload.items())


def _serialize_score_breakdown(payload: dict[str, float]) -> str:
    return "|".join(f"{key}={value:+.1f}" for key, value in payload.items())


if __name__ == "__main__":
    main()
