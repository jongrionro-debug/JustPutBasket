import json
from pathlib import Path
from types import SimpleNamespace
import sys

from switch_query.image_module.local_vlm_tagger import _coerce_json
from switch_query.image_module.preprocessing import SubprocessJsonTagger, SampleRow


def test_coerce_json_marks_invalid_model_output_for_review() -> None:
    payload = _coerce_json("not json")

    assert payload["review_needed"] is True
    assert payload["confidence_note"] == "Model output was not valid JSON"


def test_coerce_json_extracts_fenced_json_block() -> None:
    payload = _coerce_json(
        """
        Here is the result:
        ```json
        {"caption":"black wool coat","category":["coat","trousers"],"detail":["long wool coat","tailored trousers"],"color":"black","material":"wool","mood":"minimal","review_needed":false,"confidence_note":"high"}
        ```
        """
    )

    assert payload["caption"] == "black wool coat"
    assert payload["category"] == "coat|trousers"
    assert payload["detail"] == "long wool coat|tailored trousers"
    assert payload["review_needed"] is False


def test_coerce_json_extracts_object_from_wrapped_text() -> None:
    payload = _coerce_json(
        'Result follows: {"caption":"ivory silk dress","category":"dress","color":"ivory","material":"silk","mood":"romantic","review_needed":false,"confidence_note":"clear"} End.'
    )

    assert payload["category"] == "dress"
    assert payload["material"] == "silk"


def test_coerce_json_accepts_python_style_dict_output() -> None:
    payload = _coerce_json(
        "{'caption': 'navy blazer', 'category': ['blazer', 'trousers'], 'detail': ['single-breasted blazer', 'pleated trousers'], 'color': 'navy', 'material': 'wool', 'mood': 'tailored', 'review_needed': True, 'confidence_note': 'slight occlusion'}"
    )

    assert payload["caption"] == "navy blazer"
    assert payload["detail"] == "single-breasted blazer|pleated trousers"
    assert payload["review_needed"] is True


def test_coerce_json_reads_text_from_generation_result_like_object() -> None:
    payload = _coerce_json(
        SimpleNamespace(
            text=json.dumps(
                {
                    "caption": "black wool coat",
                    "category": ["coat", "shoes"],
                    "detail": ["long wool coat", "boots"],
                    "color": "black",
                    "material": "wool",
                    "mood": "minimal",
                    "review_needed": False,
                    "confidence_note": "high",
                }
            )
        )
    )

    assert payload["caption"] == "black wool coat"
    assert payload["category"] == "coat|shoes"
    assert payload["detail"] == "long wool coat|boots"
    assert payload["review_needed"] is False


def test_coerce_json_writes_raw_output_log(tmp_path: Path) -> None:
    payload = _coerce_json(
        "not json",
        image_path="/tmp/look-01.jpg",
        raw_output_log_dir=str(tmp_path),
    )

    log_files = sorted(tmp_path.iterdir())

    assert payload["review_needed"] is True
    assert len(log_files) == 1
    assert log_files[0].name.startswith("look-01-")
    assert log_files[0].name.endswith("-invalid.txt")
    assert log_files[0].read_text(encoding="utf-8") == "not json"


def test_subprocess_json_tagger_reads_structured_stdout(tmp_path: Path) -> None:
    script = tmp_path / "fake_tagger.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "print(json.dumps({",
                "  'caption': 'black wool coat',",
                "  'category': ['coat', 'trousers'],",
                "  'detail': ['long wool coat', 'tailored trousers'],",
                "  'color': 'black',",
                "  'material': 'wool',",
                "  'mood': 'minimal',",
                "  'review_needed': False,",
                "  'confidence_note': 'high'",
                "} ))",
            ]
        ),
        encoding="utf-8",
    )
    tagger = SubprocessJsonTagger([sys.executable, str(script)])
    sample = SampleRow(
        image_id="alpha:0001",
        file_path="/tmp/image.jpg",
        season_group="spring-ready-to-wear",
        year="2026",
        brand="alpha",
        source_type="collection",
        filename="0001.jpg",
        sample_reason="brand_first",
    )

    result = tagger.tag_image(sample)

    assert result.caption == "black wool coat"
    assert result.category == "coat|trousers"
    assert result.detail == "long wool coat|tailored trousers"
    assert result.mood == "minimal"
