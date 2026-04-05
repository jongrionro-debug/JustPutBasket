"""Text encoder adapters for the V2 retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from switch_query.v1.encoder import DEFAULT_SIGLIP2_MODEL, SigLIP2Encoder, SigLIP2EncoderConfig

from .models import TextEncoder


@dataclass(slots=True)
class SigLIP2TextEncoderConfig:
    model_name: str = DEFAULT_SIGLIP2_MODEL
    device: str | None = None
    batch_size: int = 8


class SigLIP2TextEncoder(TextEncoder):
    """Thin text-only adapter over the shared SigLIP2 multimodal encoder."""

    def __init__(self, config: SigLIP2TextEncoderConfig | None = None) -> None:
        self.config = config or SigLIP2TextEncoderConfig()
        self._encoder = SigLIP2Encoder(
            SigLIP2EncoderConfig(
                model_name=self.config.model_name,
                device=self.config.device,
                batch_size=self.config.batch_size,
            )
        )
        self.device = self._encoder.device

    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encoder.encode_text(texts)
