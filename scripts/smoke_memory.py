"""Live PostgreSQL + pgvector + Ollama embedding smoke test."""

import os

from research_memory_agent.embeddings import OllamaEmbedder
from research_memory_agent.models import MemoryType
from research_memory_agent.storage import PostgresMemoryStore

embedder = OllamaEmbedder(
    os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b"),
)
store = PostgresMemoryStore.connect(
    dsn=os.environ["POSTGRES_DSN"], embedder=embedder, dimensions=1024
)
try:
    store.initialize()
    memory_id = store.append_vector(
        MemoryType.SEMANTIC,
        "MemGPT gives language models tiered memory for long-running conversations.",
        metadata={"source": "smoke-test"},
    )
    matches = store.search_vector(
        MemoryType.SEMANTIC, "How can an LLM remember beyond its context window?", limit=1
    )
    assert matches and matches[0].id == memory_id
    print("MEMORY_SMOKE_OK=" + matches[0].id)
finally:
    store.close()
