from __future__ import annotations

import pytest

from switch_query.v2 import LuxiaQueryParser, LuxiaQueryParserConfig


class FakeResponse:
    def __init__(self, *, status_code=200, json_payload=None, text=""):
        self.status_code = status_code
        self._json_payload = json_payload
        self.text = text

    def json(self):
        if isinstance(self._json_payload, Exception):
            raise self._json_payload
        return self._json_payload


class FakeSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.post_calls = []

    def post(self, url, headers, json, timeout):
        self.post_calls.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": timeout,
            }
        )
        if not self.responses:
            raise AssertionError("No fake POST response configured")
        return self.responses.pop(0)


def build_feature_vocabulary() -> dict[str, dict[str, str]]:
    return {
        "category": {"coat": "coat", "gown": "dress"},
        "silhouette": {"tailored": "tailored", "sharp tailoring": "tailored"},
        "color": {"black": "black", "jet black": "black"},
        "material": {},
        "pattern": {},
        "texture": {},
        "mood": {"minimal but sharp": "minimal|sharp", "minimal": "minimal"},
        "season": {},
        "era": {},
        "detail": {},
    }


def test_luxia_query_parser_returns_validated_structured_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        [
            FakeResponse(
                json_payload={
                    "choices": [
                        {
                            "message": {
                                "content": """```json
{
  "canonical_tags": {
    "category": ["coat"],
    "color": ["black"],
    "silhouette": ["tailored"],
    "mood": ["minimal", "sharp"]
  },
  "raw_phrases": {
    "category": "coat",
    "color": "jet black",
    "silhouette": "sharp tailoring",
    "mood": "minimal but sharp"
  },
  "required_features": ["category", "color"],
  "preferred_features": ["silhouette", "mood"],
  "confidence": 0.92
}
```"""
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaQueryParser(
        feature_vocabulary=build_feature_vocabulary(),
        config=LuxiaQueryParserConfig(),
        session=session,
    )

    parsed = parser.parse(
        "Black tailored coat with minimal but sharp mood",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.canonical_tags == {
        "category": "coat",
        "color": "black",
        "silhouette": "tailored",
        "mood": "minimal|sharp",
    }
    assert parsed.raw_phrases["color"] == "jet black"
    assert parsed.required_features == ["category", "color"]
    assert parsed.preferred_features == ["silhouette", "mood"]
    assert parsed.confidence == pytest.approx(0.92)
    assert "query_text: Black tailored coat with minimal but sharp mood" in parsed.query_document
    assert session.post_calls[0]["headers"]["apikey"] == "secret-key"
    assert session.post_calls[0]["json"]["model"] == "luxia3-llm-32b-0731"
    assert session.post_calls[0]["json"]["stream"] is False


def test_luxia_query_parser_accepts_top_level_feature_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        [
            FakeResponse(
                json_payload={
                    "choices": [
                        {
                            "message": {
                                "content": """
{
  "category": ["trousers"],
  "color": "black",
  "silhouette": ["relaxed"],
  "mood": ["minimal"],
  "required_features": "category|color",
  "preferred_features": "silhouette|mood",
  "confidence": "high"
}
"""
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaQueryParser(
        feature_vocabulary=build_feature_vocabulary(),
        config=LuxiaQueryParserConfig(),
        session=session,
    )

    parsed = parser.parse(
        "black relaxed trousers with minimal mood",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.canonical_tags == {
        "category": "trousers",
        "color": "black",
        "silhouette": "relaxed",
        "mood": "minimal",
    }
    assert parsed.required_features == ["category", "color"]
    assert parsed.preferred_features == ["silhouette", "mood"]
    assert parsed.confidence == pytest.approx(0.9)


def test_luxia_query_parser_accepts_dict_feature_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        [
            FakeResponse(
                json_payload={
                    "choices": [
                        {
                            "message": {
                                "content": """
{
  "canonical_tags": {
    "category": "trousers",
    "color": "black",
    "silhouette": "relaxed",
    "mood": "minimal"
  },
  "required_features": {
    "category": true,
    "color": true
  },
  "preferred_features": {
    "features": ["silhouette", "mood"]
  },
  "confidence": 0.8
}
"""
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaQueryParser(
        feature_vocabulary=build_feature_vocabulary(),
        config=LuxiaQueryParserConfig(),
        session=session,
    )

    parsed = parser.parse(
        "black relaxed trousers with minimal mood",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.required_features == ["category", "color"]
    assert parsed.preferred_features == ["silhouette", "mood"]
    assert parsed.confidence == pytest.approx(0.8)


def test_luxia_query_parser_accepts_feature_map_nested_in_required_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        [
            FakeResponse(
                json_payload={
                    "choices": [
                        {
                            "message": {
                                "content": """
{
  "canonical_tags": ["trousers", "black", "relaxed", "minimal"],
  "raw_phrases": ["black relaxed trousers", "minimal mood"],
  "required_features": {
    "category": "trousers",
    "silhouette": "relaxed",
    "color": "black",
    "mood": "minimal"
  },
  "preferred_features": {},
  "confidence": 0.8
}
"""
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaQueryParser(
        feature_vocabulary=build_feature_vocabulary(),
        config=LuxiaQueryParserConfig(),
        session=session,
    )

    parsed = parser.parse(
        "black relaxed trousers with minimal mood",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.canonical_tags == {
        "category": "trousers",
        "silhouette": "relaxed",
        "color": "black",
        "mood": "minimal",
    }
    assert parsed.required_features == ["category", "silhouette", "color", "mood"]
    assert parsed.preferred_features == []
    assert parsed.confidence == pytest.approx(0.8)


def test_luxia_query_parser_retries_then_raises_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        [
            FakeResponse(json_payload={"choices": [{"message": {"content": "not json"}}]}),
            FakeResponse(json_payload={"choices": [{"message": {"content": "still not json"}}]}),
        ]
    )
    parser = LuxiaQueryParser(
        feature_vocabulary=build_feature_vocabulary(),
        config=LuxiaQueryParserConfig(max_retries=1),
        session=session,
    )

    with pytest.raises(RuntimeError, match="Luxia query parsing failed"):
        parser.parse(
            "Black tailored coat with minimal but sharp mood",
            stage="mood_board",
            balance_score=0.0,
        )

    assert len(session.post_calls) == 2


def test_luxia_query_parser_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUXIA_API_KEY", raising=False)
    parser = LuxiaQueryParser(
        feature_vocabulary=None,
        config=LuxiaQueryParserConfig(max_retries=0),
        session=FakeSession([]),
    )

    with pytest.raises(RuntimeError, match="LUXIA_API_KEY is required"):
        parser.parse(
            "black coat",
            stage="mood_board",
            balance_score=0.0,
        )
