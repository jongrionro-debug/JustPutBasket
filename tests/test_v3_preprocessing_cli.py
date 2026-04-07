from __future__ import annotations

import sys
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from switch_query.v3 import preprocessing_cli
from switch_query.v3.models import V3DocumentItem, V3ItemExtractionInput, V3ItemExtractionOutput


def test_extract_inputs_cli_prints_progress_and_writes_outputs(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    inputs_path = tmp_path / "item_inputs_sample.jsonl"
    output_path = tmp_path / "item_outputs_sample.jsonl"
    inputs = [
        V3ItemExtractionInput(
            image_id="look-1",
            file_path=str(tmp_path / "look-1.jpg"),
            brand="alpha",
            season_group="spring-ready-to-wear",
            detail="black jacket",
        ),
        V3ItemExtractionInput(
            image_id="look-2",
            file_path=str(tmp_path / "look-2.jpg"),
            brand="beta",
            season_group="spring-ready-to-wear",
            detail="white trousers",
        ),
    ]
    inputs_path.write_text(
        "\n".join(json.dumps(asdict(item), ensure_ascii=False) for item in inputs) + "\n",
        encoding="utf-8",
    )

    class FakeExtractor:
        def __init__(self) -> None:
            self.config = type("Config", (), {"model": "default"})()
            self.calls: list[str] = []

        def extract_items(self, extraction_input: V3ItemExtractionInput) -> V3ItemExtractionOutput:
            self.calls.append(extraction_input.image_id)
            return V3ItemExtractionOutput(
                items=[
                    V3DocumentItem(
                        item_id=f"{extraction_input.image_id}#1",
                        category="coat",
                        confidence=0.8,
                        evidence=[f"detail:{extraction_input.detail}"],
                        source="luxia_text_only",
                    )
                ],
                item_confidence=0.8,
                item_extraction_notes=[],
            )

    fake_extractor = FakeExtractor()
    monkeypatch.setattr(preprocessing_cli, "LuxiaItemExtractor", lambda: fake_extractor)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_cli.py",
            "extract-inputs",
            "--inputs",
            str(inputs_path),
            "--model",
            "gpt-4o-2024-08-06",
            "--output",
            str(output_path),
        ],
    )

    preprocessing_cli.main()

    captured = capsys.readouterr()
    stdout = captured.out
    stderr = captured.err
    payload_lines = output_path.read_text(encoding="utf-8").splitlines()

    assert fake_extractor.calls == ["look-1", "look-2"]
    assert "input_count=2" in stdout
    assert "image_transfer_mode=safe_resize" in stdout
    assert "progress [" in stderr
    assert "2/2" in stderr
    assert len(payload_lines) == 2


def test_extract_inputs_cli_persists_failure_placeholder_and_continues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    inputs_path = tmp_path / "item_inputs_sample.jsonl"
    output_path = tmp_path / "item_outputs_sample.jsonl"
    inputs = [
        V3ItemExtractionInput(
            image_id="look-1",
            file_path=str(tmp_path / "look-1.jpg"),
            brand="alpha",
            season_group="spring-ready-to-wear",
            detail="black jacket",
        ),
        V3ItemExtractionInput(
            image_id="look-2",
            file_path=str(tmp_path / "look-2.jpg"),
            brand="beta",
            season_group="spring-ready-to-wear",
            detail="white trousers",
        ),
    ]
    inputs_path.write_text(
        "\n".join(json.dumps(asdict(item), ensure_ascii=False) for item in inputs) + "\n",
        encoding="utf-8",
    )

    class FlakyExtractor:
        def __init__(self) -> None:
            self.config = type("Config", (), {"model": "default"})()
            self.calls = 0

        def extract_items(self, extraction_input: V3ItemExtractionInput) -> V3ItemExtractionOutput:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return V3ItemExtractionOutput(
                items=[
                    V3DocumentItem(
                        item_id=f"{extraction_input.image_id}#1",
                        category="coat",
                        confidence=0.8,
                        evidence=[],
                        source="luxia_text_only",
                    )
                ],
                item_confidence=0.8,
                item_extraction_notes=[],
            )

    monkeypatch.setattr(preprocessing_cli, "LuxiaItemExtractor", lambda: FlakyExtractor())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_cli.py",
            "extract-inputs",
            "--inputs",
            str(inputs_path),
            "--output",
            str(output_path),
        ],
    )

    preprocessing_cli.main()

    payload_lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(payload_lines) == 2
    first = json.loads(payload_lines[0])
    second = json.loads(payload_lines[1])
    assert first["items"][0]["category"] == "coat"
    assert second["items"] == []
    assert second["item_extraction_notes"][0] == "extraction_failed:look-2"


def test_probe_batch_cli_prints_probe_summary(monkeypatch, capsys) -> None:
    class FakeResult:
        def __init__(self, name: str, classification: str, http_status: int | None, detail: str) -> None:
            self.name = name
            self.method = "POST"
            self.url = f"https://example.com/{name}"
            self.classification = classification
            self.http_status = http_status
            self.detail = detail

    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    monkeypatch.setattr(
        preprocessing_cli,
        "run_luxia_batch_capability_probe",
        lambda api_key_env: [
            FakeResult("batches_create", "promising", 400, "missing input file"),
            FakeResult("batches_plain_post", "missing", 404, "not found"),
        ],
    )
    monkeypatch.setattr(
        preprocessing_cli,
        "has_promising_batch_probe_result",
        lambda results: any(result.classification == "promising" for result in results),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_cli.py",
            "probe-batch",
            "--api-key-env",
            "LUXIA_API_KEY",
        ],
    )

    preprocessing_cli.main()

    captured = capsys.readouterr()
    stdout = captured.out

    assert "api_key_env=LUXIA_API_KEY" in stdout
    assert "api_key_present=yes" in stdout
    assert "probe_count=2" in stdout
    assert "summary=promising_endpoints_found" in stdout
    assert "probe=batches_create" in stdout


def test_submit_batch_probe_cli_prints_attempts(monkeypatch, capsys) -> None:
    class FakeResult:
        def __init__(
            self,
            *,
            step: str,
            endpoint_name: str,
            classification: str,
            http_status: int | None,
            detail: str,
            file_id: str = "",
            batch_id: str = "",
        ) -> None:
            self.step = step
            self.endpoint_name = endpoint_name
            self.classification = classification
            self.http_status = http_status
            self.detail = detail
            self.file_id = file_id
            self.batch_id = batch_id
            self.url = f"https://example.com/{endpoint_name}"

    monkeypatch.setattr(
        preprocessing_cli,
        "run_luxia_batch_submit_probe",
        lambda api_key_env, model, endpoint_path: [
            FakeResult(
                step="upload",
                endpoint_name="files_create",
                classification="success",
                http_status=200,
                detail="ok",
                file_id="file-probe-1",
            ),
            FakeResult(
                step="create_batch",
                endpoint_name="batches_create",
                classification="success",
                http_status=200,
                detail="ok",
                file_id="file-probe-1",
                batch_id="batch-probe-1",
            ),
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocessing_cli.py",
            "submit-batch-probe",
            "--api-key-env",
            "LUXIA_API_KEY",
            "--model",
            "gpt-4o-2024-08-06",
        ],
    )

    preprocessing_cli.main()

    captured = capsys.readouterr()
    stdout = captured.out

    assert "attempt_count=2" in stdout
    assert "step=upload endpoint=files_create classification=success" in stdout
    assert "batch_id=batch-probe-1" in stdout
