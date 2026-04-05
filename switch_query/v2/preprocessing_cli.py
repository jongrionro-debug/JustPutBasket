"""CLI helpers for staged V2 archive preprocessing."""

from __future__ import annotations

import argparse

from .preprocessing import (
    CompletedBatchResult,
    DEFAULT_LOCAL_VLM_MODEL,
    DEFAULT_OPENAI_VISION_MODEL,
    DEFAULT_OUTPUT_ROOT,
    RecoveredFullTagResult,
    SubmittedBatchResult,
    build_preprocessing_paths,
    collect_full_tag_batch,
    recover_full_tag_from_logs,
    run_full_tag_stage,
    run_inventory_stage,
    run_normalize_stage,
    run_sample_first_stage,
    submit_full_tag_batch,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="V2 archive preprocessing helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument("--dataset-root", required=True)
    common_parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    common_parser.add_argument("--dataset-slug")

    paths_parser = subparsers.add_parser(
        "paths",
        parents=[common_parser],
        help="Print the fixed artifact paths for a dataset root.",
    )
    del paths_parser

    inventory_parser = subparsers.add_parser(
        "inventory",
        parents=[common_parser],
        help="Write inventory.csv for the dataset root.",
    )
    del inventory_parser

    sample_parser = subparsers.add_parser(
        "sample-first",
        parents=[common_parser],
        help="Write sample manifest, sample raw tags, sample frequency, and canonical draft.",
    )
    sample_parser.add_argument(
        "--tagger",
        choices=["blank", "mlx_vlm", "openai", "openai-sync", "subprocess"],
        default="openai-sync",
    )
    sample_parser.add_argument("--model", default=DEFAULT_LOCAL_VLM_MODEL)
    sample_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    sample_parser.add_argument("--raw-output-log-dir")
    sample_parser.add_argument("--limit", type=int)
    sample_parser.add_argument("--tagger-command", nargs="+")

    full_parser = subparsers.add_parser(
        "full-tag",
        parents=[common_parser],
        help="Write full raw tags and full frequency outputs.",
    )
    full_parser.add_argument(
        "--tagger",
        choices=["blank", "mlx_vlm", "openai", "openai-sync", "subprocess"],
        default="openai-sync",
    )
    full_parser.add_argument("--model", default=DEFAULT_LOCAL_VLM_MODEL)
    full_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    full_parser.add_argument("--raw-output-log-dir")
    full_parser.add_argument("--limit", type=int)
    full_parser.add_argument("--max-workers", type=int, default=8)
    full_parser.add_argument("--fail-fast", action="store_true")
    full_parser.add_argument("--tagger-command", nargs="+")

    full_submit_parser = subparsers.add_parser(
        "full-tag-submit",
        parents=[common_parser],
        help="Submit a full archive OpenAI batch tagging job.",
    )
    full_submit_parser.add_argument(
        "--tagger",
        choices=["openai-batch"],
        default="openai-batch",
    )
    full_submit_parser.add_argument("--model", default=DEFAULT_OPENAI_VISION_MODEL)
    full_submit_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    full_submit_parser.add_argument("--limit", type=int)

    full_collect_parser = subparsers.add_parser(
        "full-tag-collect",
        parents=[common_parser],
        help="Collect a completed full archive OpenAI batch tagging job.",
    )
    full_collect_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    full_collect_parser.add_argument("--job-id")

    recover_parser = subparsers.add_parser(
        "recover-full-from-logs",
        parents=[common_parser],
        help="Rebuild raw_tags_full.csv and frequency_full.csv from raw_output_logs_full.",
    )
    recover_parser.add_argument("--raw-output-log-dir")
    recover_parser.add_argument("--limit", type=int)

    normalize_parser = subparsers.add_parser(
        "normalize",
        parents=[common_parser],
        help="Write normalized_tags.csv from raw_tags_full.csv and canonical_mapping_final.csv.",
    )
    normalize_parser.add_argument("--raw-tags")
    normalize_parser.add_argument("--canonical-mappings")

    args = parser.parse_args()

    if args.command == "paths":
        paths = build_preprocessing_paths(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
        )
        print(
            "\n".join(
                [
                    f"dataset_root={paths.dataset_root}",
                    f"dataset_slug={paths.dataset_slug}",
                    f"root_dir={paths.root_dir}",
                    f"inventory_path={paths.inventory_path}",
                    f"sample_manifest_path={paths.sample_manifest_path}",
                    f"raw_tags_sample_path={paths.raw_tags_sample_path}",
                    f"frequency_sample_path={paths.frequency_sample_path}",
                    f"canonical_mapping_draft_path={paths.canonical_mapping_draft_path}",
                    f"raw_output_logs_sample_dir={paths.raw_output_logs_sample_dir}",
                    f"raw_tags_full_path={paths.raw_tags_full_path}",
                    f"frequency_full_path={paths.frequency_full_path}",
                    f"canonical_mapping_final_path={paths.canonical_mapping_final_path}",
                    f"raw_output_logs_full_dir={paths.raw_output_logs_full_dir}",
                    f"normalized_tags_path={paths.normalized_tags_path}",
                    f"image_file_ids_full_path={paths.image_file_ids_full_path}",
                    f"batch_input_full_path={paths.batch_input_full_path}",
                    f"batch_job_full_path={paths.batch_job_full_path}",
                    f"batch_output_full_path={paths.batch_output_full_path}",
                    f"batch_errors_full_path={paths.batch_errors_full_path}",
                ]
            )
        )
        return

    if args.command == "inventory":
        result = run_inventory_stage(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
        )
        print(
            "\n".join(
                [
                    f"inventory_path={result.paths.inventory_path}",
                    f"inventory_count={result.inventory_count}",
                    f"brand_count={result.brand_count}",
                    f"source_types={'|'.join(result.source_types)}",
                ]
            )
        )
        return

    if args.command == "sample-first":
        result = run_sample_first_stage(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
            tagger_type=args.tagger,
            model_name=args.model,
            raw_output_log_dir=args.raw_output_log_dir,
            tagger_command=args.tagger_command,
            api_key_env=args.api_key_env,
            limit=args.limit,
        )
        print(
            "\n".join(
                [
                    f"sample_manifest_path={result.paths.sample_manifest_path}",
                    f"raw_tags_sample_path={result.paths.raw_tags_sample_path}",
                    f"frequency_sample_path={result.paths.frequency_sample_path}",
                    f"canonical_mapping_draft_path={result.paths.canonical_mapping_draft_path}",
                    f"inventory_count={result.inventory_count}",
                    f"sample_count={result.sample_count}",
                    f"raw_tag_count={result.raw_tag_count}",
                    f"frequency_count={result.frequency_count}",
                    f"canonical_draft_count={result.canonical_draft_count}",
                    f"review_needed_count={result.tagging_summary.review_needed_count}",
                    f"blank_caption_count={result.tagging_summary.blank_caption_count}",
                    f"invalid_log_count={result.tagging_summary.invalid_log_count}",
                ]
            )
        )
        return

    if args.command == "full-tag":
        result = run_full_tag_stage(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
            tagger_type=args.tagger,
            model_name=args.model,
            raw_output_log_dir=args.raw_output_log_dir,
            tagger_command=args.tagger_command,
            api_key_env=args.api_key_env,
            limit=args.limit,
            max_workers=args.max_workers,
            fail_fast=args.fail_fast,
            progress_callback=_print_progress,
        )
        print(
            "\n".join(
                [
                    f"raw_tags_full_path={result.paths.raw_tags_full_path}",
                    f"frequency_full_path={result.paths.frequency_full_path}",
                    f"inventory_count={result.inventory_count}",
                    f"raw_tag_count={result.raw_tag_count}",
                    f"frequency_count={result.frequency_count}",
                    f"review_needed_count={result.tagging_summary.review_needed_count}",
                    f"blank_caption_count={result.tagging_summary.blank_caption_count}",
                    f"invalid_log_count={result.tagging_summary.invalid_log_count}",
                ]
            )
        )
        return

    if args.command == "full-tag-submit":
        result = submit_full_tag_batch(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
            model_name=args.model,
            api_key_env=args.api_key_env,
            limit=args.limit,
        )
        _print_submitted_batch_result(result)
        return

    if args.command == "full-tag-collect":
        result = collect_full_tag_batch(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
            api_key_env=args.api_key_env,
            job_id=args.job_id,
        )
        _print_completed_batch_result(result)
        return

    if args.command == "recover-full-from-logs":
        result = recover_full_tag_from_logs(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
            raw_output_log_dir=args.raw_output_log_dir,
            limit=args.limit,
        )
        _print_recovered_full_tag_result(result)
        return

    if args.command == "normalize":
        result = run_normalize_stage(
            args.dataset_root,
            output_root=args.output_root,
            dataset_slug=args.dataset_slug,
            raw_tags_path=args.raw_tags,
            canonical_mappings_path=args.canonical_mappings,
        )
        print(
            "\n".join(
                [
                    f"normalized_tags_path={result.paths.normalized_tags_path}",
                    f"raw_tag_count={result.raw_tag_count}",
                    f"normalized_count={result.normalized_count}",
                    f"rows_with_any_canonical_value={result.rows_with_any_canonical_value}",
                ]
            )
        )


def _print_submitted_batch_result(result: SubmittedBatchResult) -> None:
    print(
        "\n".join(
            [
                f"image_file_ids_full_path={result.paths.image_file_ids_full_path}",
                f"batch_input_full_path={result.paths.batch_input_full_path}",
                f"batch_job_full_path={result.paths.batch_job_full_path}",
                f"job_id={result.job_id}",
                f"input_file_id={result.input_file_id}",
                f"output_file_id={result.output_file_id or ''}",
                f"error_file_id={result.error_file_id or ''}",
                f"submitted_row_count={result.submitted_row_count}",
            ]
        )
    )


def _print_completed_batch_result(result: CompletedBatchResult) -> None:
    print(
        "\n".join(
            [
                f"batch_job_full_path={result.paths.batch_job_full_path}",
                f"batch_output_full_path={result.paths.batch_output_full_path}",
                f"batch_errors_full_path={result.paths.batch_errors_full_path}",
                f"raw_tags_full_path={result.paths.raw_tags_full_path}",
                f"frequency_full_path={result.paths.frequency_full_path}",
                f"job_id={result.job_id}",
                f"status={result.status}",
                f"raw_tag_count={result.raw_tag_count}",
                f"frequency_count={result.frequency_count}",
                f"invalid_row_count={result.invalid_row_count}",
                f"review_needed_count={result.review_needed_count}",
            ]
        )
    )


def _print_recovered_full_tag_result(result: RecoveredFullTagResult) -> None:
    print(
        "\n".join(
            [
                f"raw_tags_full_path={result.paths.raw_tags_full_path}",
                f"frequency_full_path={result.paths.frequency_full_path}",
                f"inventory_count={result.inventory_count}",
                f"recovered_count={result.recovered_count}",
                f"frequency_count={result.frequency_count}",
                f"available_parsed_log_count={result.available_parsed_log_count}",
                f"duplicate_stem_count={result.duplicate_stem_count}",
                f"review_needed_count={result.tagging_summary.review_needed_count}",
                f"blank_caption_count={result.tagging_summary.blank_caption_count}",
                f"invalid_log_count={result.tagging_summary.invalid_log_count}",
            ]
        )
    )


def _print_progress(completed: int, total: int, success_count: int, error_count: int) -> None:
    if completed == total or completed == 1 or completed % 25 == 0:
        print(
            f"progress completed={completed}/{total} "
            f"success_count={success_count} error_count={error_count}"
        )


if __name__ == "__main__":
    main()
