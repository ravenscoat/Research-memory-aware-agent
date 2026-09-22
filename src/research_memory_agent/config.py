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

    oracle_user: str = "VECTOR"
    oracle_password: str = ""
    oracle_dsn: str = "127.0.0.1:1521/FREEPDB1"
    openai_model: str = "gpt-5-mini"
    embedding_model: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    embedding_dimensions: int = 768
    context_token_limit: int = 10_000
    offload_threshold: float = 0.80
    max_iterations: int = 10
    tool_result_char_limit: int = 3_000

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            oracle_user=os.getenv("ORACLE_USER", "VECTOR"),
            oracle_password=os.getenv("ORACLE_PASSWORD", ""),
            oracle_dsn=os.getenv("ORACLE_DSN", "127.0.0.1:1521/FREEPDB1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL", "sentence-transformers/paraphrase-mpnet-base-v2"
            ),
            embedding_dimensions=_int("EMBEDDING_DIMENSIONS", 768),
            context_token_limit=_int("CONTEXT_TOKEN_LIMIT", 10_000),
            offload_threshold=_float("CONTEXT_OFFLOAD_THRESHOLD", 0.80),
            max_iterations=_int("MAX_AGENT_ITERATIONS", 10),
            tool_result_char_limit=_int("TOOL_RESULT_CHAR_LIMIT", 3_000),
        )

    def validate(self) -> None:
        if not self.oracle_password:
            raise ValueError("ORACLE_PASSWORD is required")
        if not 0 < self.offload_threshold < 1:
            raise ValueError("CONTEXT_OFFLOAD_THRESHOLD must be between 0 and 1")
        if self.embedding_dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be positive")
