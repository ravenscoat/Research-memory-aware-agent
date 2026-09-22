from __future__ import annotations

import os
from dataclasses import dataclass


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Runtime configuration loaded from environment variables."""

    postgres_dsn: str = "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = 1024
    context_token_limit: int = 10_000
    offload_threshold: float = 0.80
    max_iterations: int = 10
    tool_result_char_limit: int = 3_000

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            postgres_dsn=os.getenv(
                "POSTGRES_DSN", "postgresql://postgres:postgres@127.0.0.1:5432/postgres"
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "qwen3-embedding:0.6b"
            ),
            embedding_dimensions=_int("EMBEDDING_DIMENSIONS", 1024),
            context_token_limit=_int("CONTEXT_TOKEN_LIMIT", 10_000),
            offload_threshold=_float("CONTEXT_OFFLOAD_THRESHOLD", 0.80),
            max_iterations=_int("MAX_AGENT_ITERATIONS", 10),
            tool_result_char_limit=_int("TOOL_RESULT_CHAR_LIMIT", 3_000),
        )

    def validate(self) -> None:
        if not self.postgres_dsn:
            raise ValueError("POSTGRES_DSN is required")
        if not self.ollama_model:
            raise ValueError("OLLAMA_MODEL is required")
        if not 0 < self.offload_threshold < 1:
            raise ValueError("CONTEXT_OFFLOAD_THRESHOLD must be between 0 and 1")
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive")
