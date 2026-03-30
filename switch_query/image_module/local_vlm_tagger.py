"""Local VLM JSON tagger entrypoint for preprocessing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
DEFAULT_PROMPT = """
You are tagging one fashion runway image for retrieval preprocessing.
Return JSON only with these exact keys:
caption, category, color, material, mood, review_needed, confidence_note

Rules:
- caption: one short sentence
- category: broad parent garment category only
- color: dominant main color only
- material: main visible material only
- mood: one concise fashion mood label
- review_needed: true if image is ambiguous or uncertain
- confidence_note: short reason for confidence or uncertainty
- If uncertain, leave the field as an empty string instead of guessing
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local VLM fashion tagging")
    parser.add_argument("image_path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    args = parser.parse_args()

    payload = run_local_vlm(image_path=args.image_path, model=args.model, prompt=args.prompt)
    print(json.dumps(payload, ensure_ascii=True))


def run_local_vlm(image_path: str, model: str, prompt: str) -> dict[str, object]:
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
    return _coerce_json(output)


def _coerce_json(output: Any) -> dict[str, object]:
    if not isinstance(output, (str, bytes, bytearray)):
        output = getattr(output, "text", output)
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        payload = {
            "caption": "",
            "category": "",
            "color": "",
            "material": "",
            "mood": "",
            "review_needed": True,
            "confidence_note": "Model output was not valid JSON",
        }
    normalized = {
        "caption": str(payload.get("caption", "")).strip(),
        "category": str(payload.get("category", "")).strip(),
        "color": str(payload.get("color", "")).strip(),
        "material": str(payload.get("material", "")).strip(),
        "mood": str(payload.get("mood", "")).strip(),
        "review_needed": bool(payload.get("review_needed", False)),
        "confidence_note": str(payload.get("confidence_note", "")).strip(),
    }
    return normalized


if __name__ == "__main__":
    main()
