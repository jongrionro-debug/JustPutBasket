"""CLI entrypoint for synonym and preprocessing v1."""

from __future__ import annotations

import argparse
import csv
import sys

from .preprocessing import (
    BlankTagger,
    SubprocessJsonTagger,
    TaggingResult,
    build_frequency_rows,
    build_image_inventory,
    build_sample_manifest,
    evaluate_retrieval,
    read_canonical_mappings,
    run_rough_tagging,
    seed_canonical_mappings,
    write_csv,
    apply_canonical_mappings,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synonym and preprocessing v1 helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument("--dataset-root", required=True)
    inventory_parser.add_argument("--output", required=True)

    sample_parser = subparsers.add_parser("sample")
    sample_parser.add_argument("--dataset-root", required=True)
    sample_parser.add_argument("--output", required=True)

    tag_parser = subparsers.add_parser("tag")
    tag_parser.add_argument("--dataset-root", required=True)
    tag_parser.add_argument("--output", required=True)
    tag_parser.add_argument("--limit", type=int)
    tag_parser.add_argument(
        "--tagger",
        choices=["blank", "mlx_vlm", "subprocess"],
        default="blank",
    )
    tag_parser.add_argument("--tagger-command", nargs="+")
    tag_parser.add_argument(
        "--model",
        default="mlx-community/Qwen2-VL-2B-Instruct-4bit",
    )

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
        manifest = build_sample_manifest(inventory)
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


def _read_rows(path: str):
    from .preprocessing import RawTagRow

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [RawTagRow(**row) for row in reader]


def _read_normalized_rows(path: str):
    from .preprocessing import NormalizedTagRow

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
    return SubprocessJsonTagger(
        [
            sys.executable,
            "-m",
            "switch_query.image_module.local_vlm_tagger",
            "--model",
            args.model,
        ]
    )


if __name__ == "__main__":
    main()
