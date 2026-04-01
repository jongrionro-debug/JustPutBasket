"""CLI helpers for building and inspecting the V1 archive index."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
from pathlib import Path

from switch_query.tagging import build_image_inventory

from .encoder import SigLIP2Encoder, SigLIP2EncoderConfig
from .generator import LuxiaImageGenerator, LuxiaImageGeneratorConfig
from .index import JsonArchiveIndexStore
from .models import RetrievalMode, V1PipelineInput, V1RankedResult
from .pipeline import V1Pipeline
from .policies import V1PipelineConfig


@dataclass(slots=True)
class BuildIndexResult:
    dataset_root: str
    output_path: str
    model_name: str
    device: str
    batch_size: int
    image_count: int
    brand_count: int


@dataclass(slots=True)
class RunQueryResult:
    query_id: str
    query_text: str
    index_path: str
    retrieval_mode: RetrievalMode
    output_path: str | None
    html_output_path: str | None
    generated_image_paths: list[str]
    revised_prompt: str | None
    model_name: str
    device: str
    batch_size: int
    generated_image_count: int
    result_count: int


class _NoOpImageGenerator:
    def generate(
        self,
        query_text: str,
        count: int,
        *,
        query_id: str,
        balance_score: float,
    ) -> list[str]:
        raise RuntimeError("Synthetic image generation is not configured for this CLI.")


def build_archive_index(
    *,
    dataset_root: str,
    output_path: str,
    model_name: str,
    device: str | None,
    batch_size: int,
) -> BuildIndexResult:
    inventory_rows = build_image_inventory(dataset_root)
    if not inventory_rows:
        raise ValueError(f"No collection images found under {dataset_root}")

    encoder = SigLIP2Encoder(
        SigLIP2EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )
    index_store = JsonArchiveIndexStore(output_path)
    pipeline = V1Pipeline(
        encoder=encoder,
        image_generator=_NoOpImageGenerator(),
        index_store=index_store,
    )
    records = pipeline.build_archive_index(inventory_rows)
    return BuildIndexResult(
        dataset_root=str(Path(dataset_root).resolve()),
        output_path=str(Path(output_path).resolve()),
        model_name=model_name,
        device=encoder.device,
        batch_size=batch_size,
        image_count=len(records),
        brand_count=len({row.brand for row in inventory_rows}),
    )


def run_query(
    *,
    index_path: str,
    query_text: str,
    query_id: str,
    balance_score: float,
    retrieval_mode: RetrievalMode,
    generated_image_paths: list[str],
    output_path: str | None,
    html_output_path: str | None,
    use_luxia_generation: bool = False,
    luxia_api_key_env: str = "LUXIA_API_KEY",
    luxia_output_dir: str = "tmp/generated_refs",
    model_name: str,
    device: str | None,
    batch_size: int,
    top_k: int,
) -> RunQueryResult:
    revised_prompt: str | None = None
    resolved_generated_image_paths = list(generated_image_paths)

    if retrieval_mode in {"fusion", "image_only"} and not resolved_generated_image_paths:
        if use_luxia_generation:
            luxia_generator = LuxiaImageGenerator(
                LuxiaImageGeneratorConfig(
                    api_key_env=luxia_api_key_env,
                    output_dir=luxia_output_dir,
                )
            )
            resolved_generated_image_paths = luxia_generator.generate(
                query_text,
                1 if balance_score > -0.15 else (4 if balance_score <= -0.70 else 3),
                query_id=query_id,
                balance_score=balance_score,
            )
            if luxia_generator.generated_metadata:
                revised_prompt = luxia_generator.generated_metadata[0].revised_prompt or None
        else:
            raise ValueError(
                "generated_image_paths are required for fusion/image_only mode. "
                "Use --generated-image, or enable --use-luxia-generation."
            )

    if retrieval_mode in {"fusion", "image_only"} and not resolved_generated_image_paths:
        raise ValueError(
            "generated_image_paths are required for fusion/image_only mode. "
            "Use --generated-image, or enable --use-luxia-generation."
        )

    encoder = SigLIP2Encoder(
        SigLIP2EncoderConfig(
            model_name=model_name,
            device=device,
            batch_size=batch_size,
        )
    )
    pipeline = V1Pipeline(
        encoder=encoder,
        image_generator=_NoOpImageGenerator(),
        index_store=JsonArchiveIndexStore(index_path),
        config=V1PipelineConfig(
            top_k=top_k,
            retrieval_mode=retrieval_mode,
        ),
    )
    output = pipeline.run(
        V1PipelineInput(
            query_id=query_id,
            query_text=query_text,
            balance_score=balance_score,
            generated_image_paths=resolved_generated_image_paths,
        )
    )
    if output_path:
        write_ranked_results_csv(output_path, output.archive_results)
    if html_output_path:
        write_ranked_results_html(
            html_output_path,
            output.archive_results,
            query_text=query_text,
            retrieval_mode=retrieval_mode,
        )
    return RunQueryResult(
        query_id=query_id,
        query_text=query_text,
        index_path=str(Path(index_path).resolve()),
        retrieval_mode=retrieval_mode,
        output_path=str(Path(output_path).resolve()) if output_path else None,
        html_output_path=str(Path(html_output_path).resolve()) if html_output_path else None,
        generated_image_paths=resolved_generated_image_paths,
        revised_prompt=revised_prompt,
        model_name=model_name,
        device=encoder.device,
        batch_size=batch_size,
        generated_image_count=len(output.generated_references),
        result_count=len(output.archive_results),
    )


def write_ranked_results_csv(path: str, results: list[V1RankedResult]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query_text",
                "rank",
                "image_id",
                "final_score",
                "text_score",
                "image_score",
                "brand",
                "file_path",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "query_id": result.query_id,
                    "query_text": result.query_text,
                    "rank": result.rank,
                    "image_id": result.image_id,
                    "final_score": result.final_score,
                    "text_score": result.text_score,
                    "image_score": result.image_score,
                    "brand": result.brand,
                    "file_path": result.file_path,
                }
            )


def write_ranked_results_html(
    path: str,
    results: list[V1RankedResult],
    *,
    query_text: str,
    retrieval_mode: RetrievalMode,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for result in results:
        image_uri = Path(result.file_path).resolve().as_uri()
        cards.append(
            f"""
            <article class="card">
              <img src="{escape(image_uri)}" alt="{escape(result.image_id)}" loading="lazy">
              <div class="meta">
                <div><strong>rank</strong> {result.rank}</div>
                <div><strong>image_id</strong> {escape(result.image_id)}</div>
                <div><strong>brand</strong> {escape(result.brand)}</div>
                <div><strong>final</strong> {result.final_score:.6f}</div>
                <div><strong>text</strong> {result.text_score:.6f}</div>
                <div><strong>image</strong> {result.image_score:.6f}</div>
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
        <title>V1 Query Preview</title>
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
            grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
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
            <h1>V1 Retrieval Preview</h1>
            <div class="summary">mode: {escape(retrieval_mode)} | results: {len(results)}</div>
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
    parser = argparse.ArgumentParser(description="V1 retrieval helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index_parser = subparsers.add_parser(
        "build-index",
        help="Build archive image embeddings and write a local JSON index.",
    )
    build_index_parser.add_argument("--dataset-root", required=True)
    build_index_parser.add_argument("--output", required=True)
    build_index_parser.add_argument(
        "--model-name",
        default="google/siglip2-base-patch16-224",
    )
    build_index_parser.add_argument("--device")
    build_index_parser.add_argument("--batch-size", type=int, default=8)

    run_query_parser = subparsers.add_parser(
        "run-query",
        help="Run a query against a saved archive index.",
    )
    run_query_parser.add_argument("--index", required=True)
    run_query_parser.add_argument("--query", required=True)
    run_query_parser.add_argument("--query-id", default="query-1")
    run_query_parser.add_argument("--balance-score", type=float, default=0.0)
    run_query_parser.add_argument(
        "--retrieval-mode",
        choices=["text_only", "image_only", "fusion"],
        default="text_only",
    )
    run_query_parser.add_argument("--generated-image", action="append", default=[])
    run_query_parser.add_argument("--output")
    run_query_parser.add_argument("--use-luxia-generation", action="store_true")
    run_query_parser.add_argument("--luxia-api-key-env", default="LUXIA_API_KEY")
    run_query_parser.add_argument("--luxia-output-dir", default="tmp/generated_refs")
    run_query_parser.add_argument(
        "--model-name",
        default="google/siglip2-base-patch16-224",
    )
    run_query_parser.add_argument("--html-output")
    run_query_parser.add_argument("--device")
    run_query_parser.add_argument("--batch-size", type=int, default=8)
    run_query_parser.add_argument("--top-k", type=int, default=20)

    args = parser.parse_args()

    if args.command == "build-index":
        result = build_archive_index(
            dataset_root=args.dataset_root,
            output_path=args.output,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
        )
        print(
            "\n".join(
                [
                    f"dataset_root={result.dataset_root}",
                    f"output_path={result.output_path}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"image_count={result.image_count}",
                    f"brand_count={result.brand_count}",
                ]
            )
        )
        return

    if args.command == "run-query":
        result = run_query(
            index_path=args.index,
            query_text=args.query,
            query_id=args.query_id,
            balance_score=args.balance_score,
            retrieval_mode=args.retrieval_mode,
            generated_image_paths=args.generated_image,
            output_path=args.output,
            html_output_path=args.html_output,
            use_luxia_generation=args.use_luxia_generation,
            luxia_api_key_env=args.luxia_api_key_env,
            luxia_output_dir=args.luxia_output_dir,
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
        print(
            "\n".join(
                [
                    f"query_id={result.query_id}",
                    f"query_text={result.query_text}",
                    f"index_path={result.index_path}",
                    f"retrieval_mode={result.retrieval_mode}",
                    f"output_path={result.output_path or ''}",
                    f"html_output_path={result.html_output_path or ''}",
                    f"generated_image_count={result.generated_image_count}",
                    f"generated_image_paths={'|'.join(result.generated_image_paths)}",
                    f"revised_prompt={result.revised_prompt or ''}",
                    f"model_name={result.model_name}",
                    f"device={result.device}",
                    f"batch_size={result.batch_size}",
                    f"result_count={result.result_count}",
                ]
            )
        )


if __name__ == "__main__":
    main()
