"""SigLIP2 encoder implementation for the V1 retrieval scaffold."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Sequence

from .models import MultimodalEncoder

DEFAULT_SIGLIP2_MODEL = "google/siglip2-base-patch16-224"


@dataclass(slots=True)
class SigLIP2EncoderConfig:
    model_name: str = DEFAULT_SIGLIP2_MODEL
    device: str | None = None
    batch_size: int = 8


class SigLIP2Encoder(MultimodalEncoder):
    """Multimodal encoder backed by Hugging Face SigLIP2."""

    def __init__(self, config: SigLIP2EncoderConfig | None = None) -> None:
        self.config = config or SigLIP2EncoderConfig()
        self._torch, image_module, auto_model_cls, auto_processor_cls = self._load_runtime()
        self._image_module = image_module
        self.device = self.config.device or self._detect_device()
        self.processor = auto_processor_cls.from_pretrained(self.config.model_name)
        self.model = auto_model_cls.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(list(texts), self.config.batch_size):
            processor_kwargs: dict[str, Any] = {
                "text": batch,
                "padding": True,
                "truncation": True,
                "return_tensors": "pt",
            }
            tokenizer = getattr(self.processor, "tokenizer", None)
            model_max_length = getattr(tokenizer, "model_max_length", None)
            if isinstance(model_max_length, int) and 0 < model_max_length < 1_000_000:
                processor_kwargs["max_length"] = model_max_length
            try:
                inputs = self.processor(**processor_kwargs)
            except TypeError:
                processor_kwargs.pop("max_length", None)
                processor_kwargs.pop("truncation", None)
                inputs = self.processor(**processor_kwargs)
            vectors.extend(self._encode_features("get_text_features", inputs))
        return vectors

    def encode_image(self, image_paths: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for batch in _batched(list(image_paths), self.config.batch_size):
            images = [self._load_rgb_image(path) for path in batch]
            inputs = self.processor(images=images, return_tensors="pt")
            vectors.extend(self._encode_features("get_image_features", inputs))
        return vectors

    def _encode_features(self, method_name: str, inputs: Any) -> list[list[float]]:
        encoded_inputs = inputs.to(self.device) if hasattr(inputs, "to") else inputs
        with self._torch.no_grad():
            raw_features = getattr(self.model, method_name)(**encoded_inputs)
        if hasattr(raw_features, "pooler_output"):
            raw_features = raw_features.pooler_output
        normalized = self._torch.nn.functional.normalize(raw_features, p=2, dim=-1)
        return normalized.cpu().tolist()

    def _load_rgb_image(self, image_path: str) -> Any:
        with self._image_module.open(Path(image_path)) as image:
            return image.convert("RGB")

    def _detect_device(self) -> str:
        if self._torch.cuda.is_available():
            return "cuda"
        mps_backend = getattr(self._torch.backends, "mps", None)
        if mps_backend and mps_backend.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _load_runtime() -> tuple[Any, Any, Any, Any]:
        try:
            torch = importlib.import_module("torch")
            image_module = importlib.import_module("PIL.Image")
            transformers = importlib.import_module("transformers")
        except ImportError as exc:
            raise RuntimeError(
                "SigLIP2Encoder requires `torch`, `transformers`, and `pillow`."
            ) from exc
        return (
            torch,
            image_module,
            transformers.AutoModel,
            transformers.AutoProcessor,
        )


def _batched(items: list[str], batch_size: int) -> list[list[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]
