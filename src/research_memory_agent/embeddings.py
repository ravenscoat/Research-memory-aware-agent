from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.request import Request, urlopen


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """Generate local embeddings through Ollama's /api/embed endpoint."""

    def __init__(self, base_url: str, model_name: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        payload: dict[str, Any] = {"model": self.model_name, "input": text}
        request = Request(
            self.base_url + "/api/embed",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        vector = data["embeddings"][0]
        return [float(value) for value in vector]
