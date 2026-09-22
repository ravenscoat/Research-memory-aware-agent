from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Lazy local embedding adapter; the model is loaded only when first used."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed(self, text: str) -> list[float]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]
