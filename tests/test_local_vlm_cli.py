import json
from pathlib import Path
from types import SimpleNamespace

from switch_query.image_module.local_vlm_tagger import _coerce_json
from switch_query.image_module.preprocessing import SubprocessJsonTagger, SampleRow


def test_coerce_json_marks_invalid_model_output_for_review() -> None:
    payload = _coerce_json("not json")

    assert payload["review_needed"] is True
    assert payload["confidence_note"] == "Model output was not valid JSON"


def test_coerce_json_reads_text_from_generation_result_like_object() -> None:
    payload = _coerce_json(
        SimpleNamespace(
            text=json.dumps(
                {
                    "caption": "black wool coat",
                    "category": "coat",
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
    assert payload["category"] == "coat"
    assert payload["review_needed"] is False


def test_subprocess_json_tagger_reads_structured_stdout(tmp_path: Path) -> None:
    script = tmp_path / "fake_tagger.py"
    script.write_text(
        "\n".join(
            [
                "import json, sys",
                "print(json.dumps({",
                "  'caption': 'black wool coat',",
                "  'category': 'coat',",
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
    tagger = SubprocessJsonTagger(["python3", str(script)])
    sample = SampleRow(
        image_id="alpha:0001",
        file_path="/tmp/image.jpg",
        season_group="spring-ready-to-wear",
        brand="alpha",
        source_type="collection",
        filename="0001.jpg",
        sample_reason="brand_first",
    )

    result = tagger.tag_image(sample)

    assert result.caption == "black wool coat"
    assert result.category == "coat"
    assert result.mood == "minimal"
