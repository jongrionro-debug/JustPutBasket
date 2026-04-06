from __future__ import annotations

import pytest

from switch_query.v3 import LuxiaV3QueryParser, LuxiaV3QueryParserConfig


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


def test_luxia_v3_query_parser_returns_validated_query(monkeypatch: pytest.MonkeyPatch) -> None:
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
  "target_items": [
    {
      "target_item_id": "item_1",
      "category": "trousers",
      "color": ["black"],
      "silhouette": ["relaxed"],
      "required_attributes": ["color", "silhouette"],
      "raw_phrase": "black relaxed trousers"
    }
  ],
  "global_constraints": {
    "mood": ["minimal"]
  },
  "style_preferences": {
    "mood": ["minimal"]
  },
  "confidence": 0.92,
  "unknown_field": "ignored"
}
```"""
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaV3QueryParser(
        config=LuxiaV3QueryParserConfig(),
        session=session,
    )

    parsed = parser.parse(
        "black relaxed trousers",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.query_text == "black relaxed trousers"
    assert len(parsed.target_items) == 1
    assert parsed.target_items[0].category == "trousers"
    assert parsed.target_items[0].color == ["black"]
    assert parsed.target_items[0].silhouette == ["relaxed"]
    assert parsed.target_items[0].required_attributes == ["color", "silhouette"]
    assert parsed.global_constraints == {"mood": ["minimal"]}
    assert parsed.style_preferences == {"mood": ["minimal"]}
    assert parsed.confidence == pytest.approx(0.92)
    assert session.post_calls[0]["headers"]["apikey"] == "secret-key"
    assert session.post_calls[0]["json"]["model"] == "luxia3-llm-32b-0731"


def test_luxia_v3_query_parser_repairs_minimal_string_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
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
  "target_items": {
    "category": "black-trousers",
    "color": "black|black",
    "silhouette": ["relaxed", "relaxed"],
    "style": "minimal|minimal"
  },
  "global_constraints": {
    "mood": "clean|minimal"
  },
  "style_preferences": {
    "era": "modern"
  },
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
    parser = LuxiaV3QueryParser(
        config=LuxiaV3QueryParserConfig(),
        session=session,
    )

    parsed = parser.parse(
        "black trousers",
        stage="mood_board",
        balance_score=0.0,
    )

    assert len(parsed.target_items) == 1
    assert parsed.target_items[0].target_item_id == "item_1"
    assert parsed.target_items[0].category == "black trousers"
    assert parsed.target_items[0].raw_phrase == "black trousers"
    assert parsed.target_items[0].color == ["black"]
    assert parsed.target_items[0].silhouette == ["relaxed"]
    assert parsed.target_items[0].style_tags == ["minimal"]
    assert parsed.global_constraints == {"mood": ["clean", "minimal"]}
    assert parsed.style_preferences == {"era": ["modern"]}
    assert parsed.confidence == pytest.approx(0.9)


def test_luxia_v3_query_parser_requires_target_items(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUXIA_API_KEY", "secret-key")
    session = FakeSession(
        [
            FakeResponse(
                json_payload={
                    "choices": [
                        {
                            "message": {
                                "content": '{"global_constraints":{"mood":["minimal"]},"confidence":0.8}'
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaV3QueryParser(
        config=LuxiaV3QueryParserConfig(max_retries=0),
        session=session,
    )

    with pytest.raises(RuntimeError, match="target_items"):
        parser.parse(
            "black trousers",
            stage="mood_board",
            balance_score=0.0,
        )


def test_luxia_v3_query_parser_requires_target_item_category(monkeypatch: pytest.MonkeyPatch) -> None:
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
  "target_items": [
    {
      "target_item_id": "item_1",
      "color": ["black"],
      "raw_phrase": "black trousers"
    }
  ],
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
    parser = LuxiaV3QueryParser(
        config=LuxiaV3QueryParserConfig(max_retries=0),
        session=session,
    )

    with pytest.raises(RuntimeError, match="missing_target_category"):
        parser.parse(
            "black trousers",
            stage="mood_board",
            balance_score=0.0,
        )


def test_luxia_v3_query_parser_repairs_single_item_attributes_from_style_preferences(
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
  "target_items": [
    {
      "target_item_id": "1",
      "category": "trousers",
      "raw_phrase": "black relaxed trousers"
    }
  ],
  "global_constraints": {
    "mood": ["minimal"]
  },
  "style_preferences": {
    "color": ["black"],
    "silhouette": ["relaxed"],
    "era": ["modern"]
  },
  "confidence": 0.94
}
"""
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaV3QueryParser(
        config=LuxiaV3QueryParserConfig(max_retries=0),
        session=session,
    )

    parsed = parser.parse(
        "black relaxed trousers",
        stage="mood_board",
        balance_score=0.0,
    )

    assert parsed.target_items[0].category == "trousers"
    assert parsed.target_items[0].color == ["black"]
    assert parsed.target_items[0].silhouette == ["relaxed"]
    assert parsed.global_constraints == {"mood": ["minimal"]}
    assert parsed.style_preferences == {"era": ["modern"]}


def test_luxia_v3_query_parser_repairs_multi_item_attributes_from_raw_phrases(
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
  "target_items": [
    {
      "target_item_id": "1",
      "category": "trousers",
      "raw_phrase": "white trousers"
    },
    {
      "target_item_id": "2",
      "category": "jacket",
      "raw_phrase": "black jacket"
    }
  ],
  "confidence": 0.95
}
"""
                            }
                        }
                    ]
                }
            )
        ]
    )
    parser = LuxiaV3QueryParser(
        config=LuxiaV3QueryParserConfig(max_retries=0),
        session=session,
    )

    parsed = parser.parse(
        "white trousers with black jacket",
        stage="mood_board",
        balance_score=0.0,
    )

    assert len(parsed.target_items) == 2
    assert parsed.target_items[0].category == "trousers"
    assert parsed.target_items[0].color == ["white"]
    assert parsed.target_items[1].category == "jacket"
    assert parsed.target_items[1].color == ["black"]


def test_luxia_v3_query_parser_rejects_orphan_top_level_item_attributes(
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
  "target_items": [
    {
      "target_item_id": "item_1",
      "category": "trousers",
      "color": ["black"],
      "raw_phrase": "black trousers"
    }
  ],
  "color": ["black"],
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
    parser = LuxiaV3QueryParser(
        config=LuxiaV3QueryParserConfig(max_retries=0),
        session=session,
    )

    with pytest.raises(RuntimeError, match="orphan_global_attribute"):
        parser.parse(
            "black trousers",
            stage="mood_board",
            balance_score=0.0,
        )
