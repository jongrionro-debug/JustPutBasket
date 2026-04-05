from __future__ import annotations

import sys
from types import SimpleNamespace

from switch_query.v2 import preprocessing_cli


def test_full_tag_cli_passes_parallel_options(monkeypatch, capsys) -> None:
    captured: dict[str, object] = {}

    def fake_run_full_tag_stage(dataset_root: str, **kwargs):
        captured["dataset_root"] = dataset_root
        captured.update(kwargs)
        return SimpleNamespace(
            paths=SimpleNamespace(
                raw_tags_full_path="raw.csv",
                frequency_full_path="freq.csv",
            ),
            inventory_count=4,
            raw_tag_count=4,
            frequency_count=10,
            tagging_summary=SimpleNamespace(
                review_needed_count=1,
                blank_caption_count=0,
                invalid_log_count=0,
            ),
        )

    monkeypatch.setattr(preprocessing_cli, "run_full_tag_stage", fake_run_full_tag_stage)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_cli.py",
            "full-tag",
            "--dataset-root",
            "data/2026/spring-ready-to-wear",
            "--tagger",
            "openai-sync",
            "--model",
            "gpt-4.1-mini",
            "--max-workers",
            "8",
        ],
    )

    preprocessing_cli.main()

    output = capsys.readouterr().out
    assert captured["dataset_root"] == "data/2026/spring-ready-to-wear"
    assert captured["max_workers"] == 8
    assert captured["progress_callback"] is preprocessing_cli._print_progress
    assert "raw_tags_full_path=raw.csv" in output


def test_recover_full_from_logs_cli_prints_recovery_summary(monkeypatch, capsys) -> None:
    def fake_recover(dataset_root: str, **kwargs):
        assert dataset_root == "data/2026/spring-ready-to-wear"
        assert kwargs["limit"] == 3100
        return SimpleNamespace(
            paths=SimpleNamespace(
                raw_tags_full_path="raw.csv",
                frequency_full_path="freq.csv",
            ),
            inventory_count=11299,
            recovered_count=3100,
            frequency_count=999,
            available_parsed_log_count=3161,
            duplicate_stem_count=100,
            tagging_summary=SimpleNamespace(
                review_needed_count=30,
                blank_caption_count=0,
                invalid_log_count=0,
            ),
        )

    monkeypatch.setattr(preprocessing_cli, "recover_full_tag_from_logs", fake_recover)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_cli.py",
            "recover-full-from-logs",
            "--dataset-root",
            "data/2026/spring-ready-to-wear",
            "--limit",
            "3100",
        ],
    )

    preprocessing_cli.main()

    output = capsys.readouterr().out
    assert "recovered_count=3100" in output
    assert "duplicate_stem_count=100" in output
