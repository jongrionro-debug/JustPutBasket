"""Luxia-backed synthetic image generation for V1 retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
import mimetypes
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests


LUXIA_DALLE3_URL = "https://bridge.luxiacloud.com/image/openai/dalle3/hd/generate"


@dataclass(slots=True)
class LuxiaGeneratedImageMeta:
    query_id: str
    prompt: str
    revised_prompt: str
    source_url: str
    local_path: str


@dataclass(slots=True)
class LuxiaImageGeneratorConfig:
    api_url: str = LUXIA_DALLE3_URL
    api_key_env: str = "LUXIA_API_KEY"
    output_dir: str = "tmp/generated_refs"
    model: str = "dall-e-3"
    size: str = "1024x1024"
    quality: str = "hd"
    style: str = "vivid"
    response_format: str = "url"
    timeout_seconds: float = 120.0
    download_timeout_seconds: float = 120.0


@dataclass(slots=True)
class LuxiaImageGenerator:
    config: LuxiaImageGeneratorConfig = field(default_factory=LuxiaImageGeneratorConfig)
    session: requests.Session | None = None
    generated_metadata: list[LuxiaGeneratedImageMeta] = field(default_factory=list)

    def generate(
        self,
        query_text: str,
        count: int,
        *,
        query_id: str,
        balance_score: float,
    ) -> list[str]:
        api_key = os.environ.get(self.config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"{self.config.api_key_env} is required for Luxia image generation."
            )

        prompt = self._build_prompt(query_text, balance_score)
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "n": count,
            "size": self.config.size,
            "quality": self.config.quality,
            "style": self.config.style,
            "response_format": self.config.response_format,
        }
        response_json = self._post_json(payload, api_key)
        image_entries = self._extract_image_entries(response_json, count)

        self.generated_metadata = []
        local_paths: list[str] = []
        for index, entry in enumerate(image_entries, start=1):
            source_url = entry["url"]
            revised_prompt = str(entry.get("revised_prompt", "")).strip()
            local_path = self._download_image(source_url, query_id=query_id, index=index)
            self.generated_metadata.append(
                LuxiaGeneratedImageMeta(
                    query_id=query_id,
                    prompt=prompt,
                    revised_prompt=revised_prompt,
                    source_url=source_url,
                    local_path=local_path,
                )
            )
            local_paths.append(local_path)

        if len(local_paths) != count:
            raise RuntimeError(
                f"Expected {count} generated images but downloaded {len(local_paths)}."
            )
        return local_paths

    def _build_prompt(self, query_text: str, balance_score: float) -> str:
        if balance_score <= -0.15:
            suffix = "Produce diverse fashion directions with varied silhouette, styling, and mood."
        else:
            suffix = "Produce a tightly aligned fashion reference faithful to the brief."
        return f"{query_text}\n{suffix}"

    def _post_json(self, payload: dict[str, object], api_key: str) -> Any:
        client = self.session or requests.Session()
        try:
            response = client.post(
                self.config.api_url,
                headers={
                    "apikey": api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"Luxia request failed: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"Luxia request failed with status {response.status_code}: {response.text}"
            )

        try:
            return response.json()
        except ValueError as exc:
            raise RuntimeError("Luxia response was not valid JSON.") from exc

    def _extract_image_entries(
        self,
        payload: Any,
        expected_count: int,
    ) -> list[dict[str, Any]]:
        if isinstance(payload, dict) and "url" in payload:
            entries = [payload]
        elif isinstance(payload, list):
            entries = [item for item in payload if isinstance(item, dict)]
        else:
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, list):
                raise RuntimeError("Luxia response did not contain image urls.")
            entries = [item for item in data if isinstance(item, dict)]

        urls = [entry for entry in entries if entry.get("url")]
        if len(urls) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} image urls but received {len(urls)}."
            )
        return urls

    def _download_image(self, url: str, *, query_id: str, index: int) -> str:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        client = self.session or requests.Session()
        try:
            response = client.get(url, timeout=self.config.download_timeout_seconds)
        except requests.RequestException as exc:
            raise RuntimeError(f"Failed to download generated image from {url}: {exc}") from exc

        if response.status_code >= 400:
            raise RuntimeError(
                f"Failed to download generated image from {url}: "
                f"status {response.status_code}"
            )

        suffix = self._infer_suffix(url, response.headers.get("Content-Type", ""))
        destination = output_dir / f"{query_id}_gen_{index:02d}{suffix}"
        destination.write_bytes(response.content)
        return str(destination.resolve())

    @staticmethod
    def _infer_suffix(url: str, content_type: str) -> str:
        path_suffix = Path(urlparse(url).path).suffix.lower()
        if path_suffix:
            return path_suffix
        guessed = mimetypes.guess_extension(content_type.split(";", 1)[0].strip())
        return guessed or ".png"
