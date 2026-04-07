"""CLI helpers for V3 item extraction preparation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .batch_probe import (
    has_promising_batch_probe_result,
    run_luxia_batch_capability_probe,
    run_luxia_batch_submit_probe,
)
from .preprocessing import (
    DEFAULT_OUTPUT_ROOT,
    append_item_extraction_output_jsonl,
    build_item_extraction_inputs,
    build_preprocessing_paths,
    slice_item_extraction_inputs,
    read_item_extraction_inputs_jsonl,
    read_item_extraction_outputs_jsonl,
    read_normalized_tag_rows,
    merge_item_inputs_and_outputs,
    write_archive_documents_jsonl,
    write_item_extraction_inputs_jsonl,
)
from .item_extractor import LuxiaItemExtractor
from .models import V3ItemExtractionOutput


def _print_progress(*, completed: int, total: int) -> None:
    if total <= 0:
        return
    bar_width = 24
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(bar_width * ratio)
    bar = "#" * filled + "-" * (bar_width - filled)
    message = f"\rprogress [{bar}] {completed}/{total} ({ratio * 100:5.1f}%)"
    end = "\n" if completed >= total else ""
    print(message, end=end, file=sys.stderr, flush=True)


def _build_failed_output(*, image_id: str, error: Exception) -> V3ItemExtractionOutput:
    return V3ItemExtractionOutput(
        items=[],
        item_confidence=0.0,
        item_extraction_notes=[
            f"extraction_failed:{image_id}",
            f"error:{type(error).__name__}:{error}",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="V3 item extraction preparation helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    paths_parser = subparsers.add_parser(
        "paths",
        help="Print the fixed V3 preprocessing paths for a normalized tag file.",
    )
    paths_parser.add_argument("--normalized-tags", required=True)
    paths_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    build_inputs_parser = subparsers.add_parser(
        "build-inputs",
        help="Build V3 item extraction input JSONL from normalized_tags.csv.",
    )
    build_inputs_parser.add_argument("--normalized-tags", required=True)
    build_inputs_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    build_inputs_parser.add_argument(
        "--mode",
        choices=["sample", "full"],
        default="sample",
    )
    build_inputs_parser.add_argument(
        "--extraction-mode",
        choices=["text_only", "image_assisted"],
        default="text_only",
    )
    build_inputs_parser.add_argument("--offset", type=int, default=0)
    build_inputs_parser.add_argument("--limit", type=int)
    build_inputs_parser.add_argument("--output")

    extract_parser = subparsers.add_parser(
        "extract-inputs",
        help="Run Luxia item extraction over a prepared input JSONL.",
    )
    extract_parser.add_argument("--inputs", required=True)
    extract_parser.add_argument("--output")
    extract_parser.add_argument("--model", default="gpt-4o-2024-08-06")
    extract_parser.add_argument("--offset", type=int, default=0)
    extract_parser.add_argument("--limit", type=int)
    extract_parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing output JSONL and skip already-written rows.",
    )
    extract_parser.add_argument(
        "--image-transfer-mode",
        choices=["raw", "safe_resize"],
        default="safe_resize",
    )
    extract_parser.add_argument("--max-image-edge", type=int, default=1024)
    extract_parser.add_argument("--jpeg-quality", type=int, default=85)

    probe_parser = subparsers.add_parser(
        "probe-batch",
        help="Probe likely Luxia OpenAI-style batch endpoints and print their responses.",
    )
    probe_parser.add_argument("--api-key-env", default="LUXIA_API_KEY")

    submit_probe_parser = subparsers.add_parser(
        "submit-batch-probe",
        help="Try a real 1-line Luxia batch upload and batch create request.",
    )
    submit_probe_parser.add_argument("--api-key-env", default="LUXIA_API_KEY")
    submit_probe_parser.add_argument("--model", default="gpt-4o-2024-08-06")
    submit_probe_parser.add_argument("--endpoint-path", default="/v1/chat/completions")

    merge_parser = subparsers.add_parser(
        "merge-documents",
        help="Merge prepared inputs and extraction outputs into V3 archive documents JSONL.",
    )
    merge_parser.add_argument("--inputs", required=True)
    merge_parser.add_argument("--outputs", required=True)
    merge_parser.add_argument("--output")

    args = parser.parse_args()

    if args.command == "paths":
        paths = build_preprocessing_paths(
            args.normalized_tags,
            output_root=args.output_root,
        )
        print(
            "\n".join(
                [
                    f"normalized_tags_path={paths.normalized_tags_path}",
                    f"root_dir={paths.root_dir}",
                    f"item_inputs_sample_path={paths.item_inputs_sample_path}",
                    f"item_inputs_full_path={paths.item_inputs_full_path}",
                    f"item_outputs_sample_path={paths.item_outputs_sample_path}",
                    f"item_outputs_full_path={paths.item_outputs_full_path}",
                    f"item_enriched_documents_sample_path={paths.item_enriched_documents_sample_path}",
                    f"item_enriched_documents_full_path={paths.item_enriched_documents_full_path}",
                ]
            )
        )
        return

    if args.command == "build-inputs":
        paths = build_preprocessing_paths(
            args.normalized_tags,
            output_root=args.output_root,
        )
        rows = read_normalized_tag_rows(args.normalized_tags)
        inputs = build_item_extraction_inputs(
            rows,
            extraction_mode=args.extraction_mode,
            offset=args.offset,
            limit=args.limit,
        )
        output_path = args.output or (
            paths.item_inputs_sample_path if args.mode == "sample" else paths.item_inputs_full_path
        )
        write_item_extraction_inputs_jsonl(output_path, inputs)
        print(
            "\n".join(
                [
                    f"normalized_tags_path={paths.normalized_tags_path}",
                    f"output_path={output_path}",
                    f"mode={args.mode}",
                    f"extraction_mode={args.extraction_mode}",
                    f"offset={args.offset}",
                    f"input_count={len(inputs)}",
                ]
            )
        )
        return

    if args.command == "extract-inputs":
        inputs = read_item_extraction_inputs_jsonl(args.inputs)
        selected_inputs = slice_item_extraction_inputs(
            inputs,
            offset=args.offset,
            limit=args.limit,
        )
        extractor = LuxiaItemExtractor()
        extractor.config.model = args.model
        extractor.config.image_transfer_mode = args.image_transfer_mode
        extractor.config.max_image_edge = args.max_image_edge
        extractor.config.jpeg_quality = args.jpeg_quality
        output_path = args.output or str(Path(args.inputs).with_name("item_outputs_sample.jsonl"))
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        resume_count = 0
        if args.resume and output_file.exists():
            resume_count = sum(1 for line in output_file.open(encoding="utf-8") if line.strip())
        else:
            output_file.write_text("", encoding="utf-8")
        if resume_count:
            selected_inputs = selected_inputs[resume_count:]
        outputs = []
        total_inputs = len(selected_inputs)
        if total_inputs:
            _print_progress(completed=0, total=total_inputs)
        for index, item in enumerate(selected_inputs, start=1):
            try:
                output = extractor.extract_items(item)
            except Exception as exc:
                print(
                    f"\nwarning: extraction failed for {item.image_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                output = _build_failed_output(image_id=item.image_id, error=exc)
            outputs.append(output)
            append_item_extraction_output_jsonl(output_path, output)
            _print_progress(completed=index, total=total_inputs)
        print(
            "\n".join(
                [
                    f"inputs_path={args.inputs}",
                    f"output_path={output_path}",
                    f"model={args.model}",
                    f"image_transfer_mode={args.image_transfer_mode}",
                    f"max_image_edge={args.max_image_edge}",
                    f"jpeg_quality={args.jpeg_quality}",
                    f"offset={args.offset}",
                    f"resume={'yes' if args.resume else 'no'}",
                    f"resume_skipped={resume_count}",
                    f"input_count={len(selected_inputs)}",
                ]
            )
        )
        return

    if args.command == "probe-batch":
        results = run_luxia_batch_capability_probe(api_key_env=args.api_key_env)
        api_key_present = "yes" if os.environ.get(args.api_key_env) else "no"
        summary = "promising_endpoints_found" if has_promising_batch_probe_result(results) else "no_promising_endpoints"
        print(
            "\n".join(
                [
                    f"api_key_env={args.api_key_env}",
                    f"api_key_present={api_key_present}",
                    f"probe_count={len(results)}",
                    f"summary={summary}",
                ]
                + [
                    (
                        f"probe={result.name} "
                        f"method={result.method} "
                        f"classification={result.classification} "
                        f"http_status={result.http_status if result.http_status is not None else 'none'} "
                        f"url={result.url} "
                        f"detail={result.detail}"
                    )
                    for result in results
                ]
            )
        )
        return

    if args.command == "submit-batch-probe":
        results = run_luxia_batch_submit_probe(
            api_key_env=args.api_key_env,
            model=args.model,
            endpoint_path=args.endpoint_path,
        )
        print(
            "\n".join(
                [
                    f"api_key_env={args.api_key_env}",
                    f"model={args.model}",
                    f"endpoint_path={args.endpoint_path}",
                    f"attempt_count={len(results)}",
                ]
                + [
                    (
                        f"step={result.step} "
                        f"endpoint={result.endpoint_name} "
                        f"classification={result.classification} "
                        f"http_status={result.http_status if result.http_status is not None else 'none'} "
                        f"url={result.url} "
                        f"file_id={result.file_id or '-'} "
                        f"batch_id={result.batch_id or '-'} "
                        f"detail={result.detail}"
                    )
                    for result in results
                ]
            )
        )
        return

    if args.command == "merge-documents":
        inputs = read_item_extraction_inputs_jsonl(args.inputs)
        outputs = read_item_extraction_outputs_jsonl(args.outputs)
        documents = merge_item_inputs_and_outputs(inputs, outputs)
        output_path = args.output or str(Path(args.outputs).with_name("item_enriched_documents_sample.jsonl"))
        write_archive_documents_jsonl(output_path, documents)
        print(
            "\n".join(
                [
                    f"inputs_path={args.inputs}",
                    f"outputs_path={args.outputs}",
                    f"output_path={output_path}",
                    f"document_count={len(documents)}",
                ]
            )
        )


if __name__ == "__main__":
    main()
