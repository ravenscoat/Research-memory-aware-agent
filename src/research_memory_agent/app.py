from __future__ import annotations

from dataclasses import dataclass

from .agent import ResearchAgent
from .config import Settings
from .context import ContextAssembler
from .embeddings import OllamaEmbedder
from .llm import OllamaChatModel
from .memory import MemoryManager
from .storage import PostgresMemoryStore
from .tools import ToolRegistry, register_research_tools


@dataclass
class Application:
    agent: ResearchAgent
    store: PostgresMemoryStore
    tools: ToolRegistry

    def close(self) -> None:
        self.store.close()


def build_application(settings: Settings | None = None) -> Application:
    settings = settings or Settings.from_env()
    settings.validate()
    embedder = OllamaEmbedder(settings.ollama_base_url, settings.embedding_model)
    store = PostgresMemoryStore.connect(
        dsn=settings.postgres_dsn,
        embedder=embedder,
        dimensions=settings.embedding_dimensions,
    )
    store.initialize()
    model = OllamaChatModel(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
    )
    memory = MemoryManager(store, model)
    tools = ToolRegistry(memory)
    register_research_tools(tools, memory)
    context = ContextAssembler(
        memory,
        token_limit=settings.context_token_limit,
        offload_threshold=settings.offload_threshold,
    )
    agent = ResearchAgent(
        model=model,
        memory=memory,
        context=context,
        tools=tools,
        max_iterations=settings.max_iterations,
        tool_result_char_limit=settings.tool_result_char_limit,
    )
    return Application(agent=agent, store=store, tools=tools)
