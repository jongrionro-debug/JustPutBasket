from __future__ import annotations

from dataclasses import dataclass

import pytest

from switch_query.v1.encoder import SigLIP2Encoder, SigLIP2EncoderConfig, _batched


class FakeTensor:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[list[float]]:
        return self.rows


class FakeOutputWithPooling:
    def __init__(self, rows: list[list[float]]) -> None:
        self.pooler_output = FakeTensor(rows)


class FakeBatch(dict):
    def __init__(self, payload: dict[str, object]) -> None:
        super().__init__(payload)
        self.device: str | None = None

    def to(self, device: str) -> "FakeBatch":
        self.device = device
        return self


class FakeProcessor:
    def __init__(self) -> None:
        self.text_calls: list[list[str]] = []
        self.image_calls: list[list[object]] = []

    @classmethod
    def from_pretrained(cls, model_name: str) -> "FakeProcessor":
        return cls()

    def __call__(self, *, text=None, images=None, padding=None, return_tensors=None):
        if text is not None:
            self.text_calls.append(list(text))
            return FakeBatch({"text": list(text)})
        self.image_calls.append(list(images))
        return FakeBatch({"images": list(images)})


class FakeModel:
    def __init__(self) -> None:
        self.device: str | None = None
        self.is_eval = False

    @classmethod
    def from_pretrained(cls, model_name: str) -> "FakeModel":
        return cls()

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> None:
        self.is_eval = True

    def get_text_features(self, **inputs):
        rows = []
        for text in inputs["text"]:
            rows.append([float(len(text)), 0.0])
        return FakeOutputWithPooling(rows)

    def get_image_features(self, **inputs):
        rows = []
        for image in inputs["images"]:
            rows.append([float(image.width), float(image.height)])
        return FakeOutputWithPooling(rows)


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


@dataclass
class FakeImage:
    width: int
    height: int
    mode: str = "RGB"

    def convert(self, mode: str) -> "FakeImage":
        return FakeImage(width=self.width, height=self.height, mode=mode)


class FakeOpenedImage:
    def __init__(self, image: FakeImage) -> None:
        self.image = image

    def __enter__(self) -> FakeImage:
        return self.image

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeImageModule:
    def __init__(self, mapping: dict[str, FakeImage]) -> None:
        self.mapping = mapping

    def open(self, path) -> FakeOpenedImage:
        return FakeOpenedImage(self.mapping[str(path)])


class FakeFunctional:
    @staticmethod
    def normalize(tensor: FakeTensor, p: int, dim: int) -> FakeTensor:
        return tensor


class FakeTorch:
    class cuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class backends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return False

    class nn:
        functional = FakeFunctional()

    @staticmethod
    def no_grad() -> FakeNoGrad:
        return FakeNoGrad()


def test_batched_chunks_items() -> None:
    assert _batched(["a", "b", "c", "d", "e"], 2) == [["a", "b"], ["c", "d"], ["e"]]


def test_batched_rejects_non_positive_batch_size() -> None:
    with pytest.raises(ValueError):
        _batched(["a"], 0)


def test_siglip2_encoder_encodes_text_and_images(monkeypatch) -> None:
    image_module = FakeImageModule(
        {
            "look-1.jpg": FakeImage(width=10, height=20),
            "look-2.jpg": FakeImage(width=30, height=40),
        }
    )

    monkeypatch.setattr(
        SigLIP2Encoder,
        "_load_runtime",
        staticmethod(lambda: (FakeTorch(), image_module, FakeModel, FakeProcessor)),
    )

    encoder = SigLIP2Encoder(SigLIP2EncoderConfig(device="cpu", batch_size=2))

    text_vectors = encoder.encode_text(["coat", "dress"])
    image_vectors = encoder.encode_image(["look-1.jpg", "look-2.jpg"])

    assert text_vectors == [[4.0, 0.0], [5.0, 0.0]]
    assert image_vectors == [[10.0, 20.0], [30.0, 40.0]]
    assert encoder.device == "cpu"
    assert encoder.model.is_eval is True


def test_siglip2_encoder_prefers_mps_when_available(monkeypatch) -> None:
    class FakeTorchWithMps(FakeTorch):
        class backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return True

    monkeypatch.setattr(
        SigLIP2Encoder,
        "_load_runtime",
        staticmethod(lambda: (FakeTorchWithMps(), FakeImageModule({}), FakeModel, FakeProcessor)),
    )

    encoder = SigLIP2Encoder(SigLIP2EncoderConfig())

    assert encoder.device == "mps"
