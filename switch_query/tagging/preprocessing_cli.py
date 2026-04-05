"""CLI entrypoint for synonym and preprocessing v1."""

from __future__ import annotations

import argparse
import csv
import sys

from .preprocessing import (
    BlankTagger,
    NormalizedTagRow,
    RawTagRow,
    SubprocessJsonTagger,
    apply_canonical_mappings,
    build_frequency_rows,
    build_full_manifest,
    build_image_inventory,
    build_sample_manifest,
    evaluate_retrieval,
    read_canonical_mappings,
    run_rough_tagging,
    seed_canonical_mappings,
    write_csv,
)
from .openai_vlm_tagger import DEFAULT_OPENAI_VISION_MODEL, OpenAIJsonTagger


def main() -> None:
    parser = argparse.ArgumentParser(description="Synonym and preprocessing v1 helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dataset_root_help = (
        "Path to the archive root. Supports roots like data, data/2026, "
        "or data/2026/spring-ready-to-wear."
    )

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--dataset-root", required=True, help=dataset_root_help)
    inventory_parser.add_argument("--output", required=True)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--dataset-root", required=True, help=dataset_root_help)
    sample_parser.add_argument("--output", required=True)

    tag_parser = subparsers.add_parser("tag")
    tag_parser.add_argument("--dataset-root", required=True, help=dataset_root_help)
    tag_parser.add_argument("--output", required=True)
    tag_parser.add_argument("--limit", type=int)
    tag_parser.add_argument(
        "--selection",
        choices=["sample", "all"],
        default="sample",
        help="Use brand first/last samples or tag every image in the dataset root.",
    )
    tag_parser.add_argument(
        "--tagger",
        choices=["blank", "mlx_vlm", "openai", "openai-sync", "subprocess"],
        default="blank",
    )
    tag_parser.add_argument("--tagger-command", nargs="+")
    tag_parser.add_argument(
        "--model",
        default="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    )
    tag_parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    tag_parser.add_argument("--raw-output-log-dir")

    frequency_parser = subparsers.add_parser("frequency")
    frequency_parser.add_argument("--raw-tags", required=True)
    frequency_parser.add_argument("--output", required=True)
    frequency_parser.add_argument("--seed-canonical-output")

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--raw-tags", required=True)
    normalize_parser.add_argument("--canonical-mappings", required=True)
    normalize_parser.add_argument("--output", required=True)

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--raw-tags", required=True)
    eval_parser.add_argument("--normalized-tags", required=True)
    eval_parser.add_argument("--output", required=True)

    args = parser.parse_args()

    if args.command == "inventory":
        rows = build_image_inventory(args.dataset_root)
        write_csv(args.output, rows)
        return

    if args.command == "sample":
        inventory = build_image_inventory(args.dataset_root)
        rows = build_sample_manifest(inventory)
        write_csv(args.output, rows)
        return

    if args.command == "tag":
        inventory = build_image_inventory(args.dataset_root)
        manifest = (
            build_full_manifest(inventory)
            if args.selection == "all"
            else build_sample_manifest(inventory)
        )
        tagger = _build_tagger(args)
        rows = run_rough_tagging(manifest, tagger, limit=args.limit)
        write_csv(args.output, rows)
        return

    if args.command == "frequency":
        raw_rows = _read_rows(args.raw_tags)
        freq_rows = build_frequency_rows(raw_rows)
        write_csv(args.output, freq_rows)
        if args.seed_canonical_output:
            write_csv(args.seed_canonical_output, seed_canonical_mappings(freq_rows))
        return

    if args.command == "normalize":
        raw_rows = _read_rows(args.raw_tags)
        mappings = read_canonical_mappings(args.canonical_mappings)
        normalized = apply_canonical_mappings(raw_rows, mappings)
        write_csv(args.output, normalized)
        return

    if args.command == "eval":
        raw_rows = _read_rows(args.raw_tags)
        normalized_rows = _read_normalized_rows(args.normalized_tags)
        logs = evaluate_retrieval(raw_rows, normalized_rows)
        write_csv(args.output, logs)


def _read_rows(path: str) -> list[RawTagRow]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [RawTagRow(**row) for row in reader]


def _read_normalized_rows(path: str) -> list[NormalizedTagRow]:
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [NormalizedTagRow(**row) for row in reader]


def _build_tagger(args) -> BlankTagger | SubprocessJsonTagger:
    if args.tagger == "blank":
        return BlankTagger()
    if args.tagger == "subprocess":
        if not args.tagger_command:
            raise ValueError("--tagger-command is required for --tagger subprocess")
        return SubprocessJsonTagger(args.tagger_command)
    if args.tagger in {"openai", "openai-sync"}:
        return OpenAIJsonTagger(
            model=(
                DEFAULT_OPENAI_VISION_MODEL
                if args.model == "mlx-community/Qwen2-VL-2B-Instruct-4bit"
                else args.model
            ),
            api_key_env=args.api_key_env,
            raw_output_log_dir=args.raw_output_log_dir,
        )
    return SubprocessJsonTagger(
        [
            sys.executable,
            "-m",
            "switch_query.tagging.local_vlm_tagger",
            "--model",
            args.model,
            *(
                ["--raw-output-log-dir", args.raw_output_log_dir]
                if args.raw_output_log_dir
                else []
            ),
        ]
    )


if __name__ == "__main__":
    main()
