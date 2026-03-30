"""Local VLM JSON tagger entrypoint for preprocessing."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
DEFAULT_PROMPT = """
You are tagging one fashion runway image for retrieval preprocessing.
Return exactly one JSON object and nothing else.
Do not add markdown, code fences, commentary, prefixes, suffixes, or explanations.
Use these exact keys only:
"caption", "category", "detail", "color", "material", "mood", "review_needed", "confidence_note"

Rules:
- Use valid JSON with double quotes for every key and string value.
- Use true or false for "review_needed", never a string.
- caption: one short sentence
- category: JSON array of all clearly visible broad garment categories, ordered by salience
- detail: JSON array of detailed item labels for the visible fashion items, ordered by salience
- color: dominant main color only
- material: main visible material only
- mood: one concise fashion mood label
- review_needed: true if image is ambiguous or uncertain
- confidence_note: short reason for confidence or uncertainty
- If uncertain, use [] for category/detail and an empty string for other uncertain fields instead of guessing.

Output example:
{"caption":"black wool coat with wide-leg trousers on runway","category":["coat","trousers","shoes"],"detail":["long wool coat","wide-leg trousers","heels"],"color":"black","material":"wool","mood":"minimal","review_needed":false,"confidence_note":"high confidence"}
""".strip()
JSON_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local VLM fashion tagging")
    parser.add_argument("image_path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-output-log-dir")
    args = parser.parse_args()

    payload = run_local_vlm(
        image_path=args.image_path,
        model=args.model,
        prompt=args.prompt,
        raw_output_log_dir=args.raw_output_log_dir,
    )
    print(json.dumps(payload, ensure_ascii=True))


def run_local_vlm(
    image_path: str,
    model: str,
    prompt: str,
    raw_output_log_dir: str | None = None,
) -> dict[str, object]:
    try:
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config
    except ImportError as exc:
        raise RuntimeError(
            "mlx-vlm is not installed. Install it before using local_vlm_tagger."
        ) from exc

    image = [str(Path(image_path).resolve())]
    loaded_model, processor = load(model)
    config = load_config(model)
    formatted_prompt = apply_chat_template(
        processor,
        config,
        prompt,
        num_images=len(image),
    )
    output = generate(
        loaded_model,
        processor,
        formatted_prompt,
        image,
        verbose=False,
        max_tokens=300,
        temperature=0.0,
    )
    return _coerce_json(
        output,
        image_path=image_path,
        raw_output_log_dir=raw_output_log_dir,
    )


def _coerce_json(
    output: Any,
    image_path: str | None = None,
    raw_output_log_dir: str | None = None,
) -> dict[str, object]:
    raw_text = _to_text(output)
    payload = _parse_payload(raw_text)
    if raw_output_log_dir:
        _write_raw_output_log(
            Path(raw_output_log_dir),
            image_path=image_path,
            raw_text=raw_text,
            status="invalid" if payload is None else "parsed",
        )
    if payload is None:
        payload = {
            "caption": "",
            "category": "",
            "detail": "",
            "color": "",
            "material": "",
            "mood": "",
            "review_needed": True,
            "confidence_note": "Model output was not valid JSON",
        }
    normalized = {
        "caption": str(payload.get("caption", "")).strip(),
        "category": _coerce_multi_value(payload.get("category", "")),
        "detail": _coerce_multi_value(payload.get("detail", "")),
        "color": str(payload.get("color", "")).strip(),
        "material": str(payload.get("material", "")).strip(),
        "mood": str(payload.get("mood", "")).strip(),
        "review_needed": _coerce_bool(payload.get("review_needed", False)),
        "confidence_note": str(payload.get("confidence_note", "")).strip(),
    }
    return normalized


def _parse_payload(raw_text: str) -> dict[str, object] | None:
    for candidate in _candidate_json_strings(raw_text):
        parsed = _load_json_dict(candidate)
        if parsed is not None:
            return parsed
        parsed = _load_python_dict(candidate)
        if parsed is not None:
            return parsed
        cleaned = _clean_candidate(candidate)
        if cleaned != candidate:
            parsed = _load_json_dict(cleaned)
            if parsed is not None:
                return parsed
            parsed = _load_python_dict(cleaned)
            if parsed is not None:
                return parsed
    return None


def _candidate_json_strings(raw_text: str) -> list[str]:
    cleaned = raw_text.strip()
    candidates: list[str] = []
    if cleaned:
        candidates.append(cleaned)

    for block in JSON_CODE_BLOCK_RE.findall(raw_text):
        block_text = block.strip()
        if block_text:
            candidates.append(block_text)

    balanced = _extract_balanced_object(cleaned)
    if balanced:
        candidates.append(balanced)

    unique_candidates: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    return unique_candidates


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    string_char = ""
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == string_char:
                in_string = False
            continue

        if char in {'"', "'"}:
            in_string = True
            string_char = char
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _load_json_dict(text: str) -> dict[str, object] | None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _load_python_dict(text: str) -> dict[str, object] | None:
    try:
        payload = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _clean_candidate(text: str) -> str:
    cleaned = text.strip()
    cleaned = (
        cleaned.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _to_text(output: Any) -> str:
    if not isinstance(output, (str, bytes, bytearray)):
        output = getattr(output, "text", output)
    if isinstance(output, (bytes, bytearray)):
        return bytes(output).decode("utf-8", errors="replace")
    if isinstance(output, str):
        return output
    return str(output)


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _coerce_multi_value(value: object) -> str:
    if isinstance(value, str):
        tokens = re.split(r"[|,\n;]+", value)
        candidates = [token.strip() for token in tokens]
    elif isinstance(value, (list, tuple, set)):
        candidates = [str(item).strip() for item in value]
    else:
        candidates = [str(value).strip()]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return "|".join(normalized)


def _write_raw_output_log(
    log_dir: Path,
    *,
    image_path: str | None,
    raw_text: str,
    status: str,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    image_name = Path(image_path).stem if image_path else "unknown"
    digest_source = image_path or raw_text
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:10]
    log_path = log_dir / f"{image_name}-{digest}-{status}.txt"
    log_path.write_text(raw_text, encoding="utf-8")


if __name__ == "__main__":
    main()
