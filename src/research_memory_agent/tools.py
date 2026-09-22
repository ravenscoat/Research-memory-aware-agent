from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

from pypdf import PdfReader

from .models import ToolDefinition


class ToolRegistry:
    """Executable tool registry whose descriptions are themselves searchable memory."""

    def __init__(self, memory: Any) -> None:
        self.memory = memory
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition, *, persist: bool = True) -> None:
        self._tools[tool.name] = tool
        if persist:
            self.memory.store.upsert_tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )

    def persist_all(self) -> None:
        for tool in self._tools.values():
            self.memory.store.upsert_tool(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            )

    def select(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        names = self.memory.select_tools(query, limit=limit)
        selected = [self._tools[name] for name in names if name in self._tools]
        # New databases can have no tool vectors during first registration/testing.
        if not selected:
            selected = list(self._tools.values())[:limit]
        return [tool.openai_schema() for tool in selected]

    def execute(self, name: str, arguments: dict[str, Any], *, thread_id: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: tool {name!r} is not registered"
        args = dict(arguments or {})
        if name == "summarize_and_store":
            args.setdefault("thread_id", thread_id)
        return str(tool.handler(**args) or "Done")


def register_research_tools(registry: ToolRegistry, memory: Any) -> None:
    registry.register(
        ToolDefinition(
            name="search_arxiv",
            description="Search arXiv for research papers and return titles, IDs, authors, and abstracts.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            handler=search_arxiv,
        )
    )
    registry.register(
        ToolDefinition(
            name="fetch_arxiv_paper",
            description="Download an arXiv paper PDF by ID and extract bounded plain text.",
            parameters={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "max_characters": {
                        "type": "integer",
                        "minimum": 1000,
                        "maximum": 100000,
                    },
                },
                "required": ["arxiv_id"],
                "additionalProperties": False,
            },
            handler=fetch_arxiv_paper,
        )
    )
    registry.register(
        ToolDefinition(
            name="save_to_knowledge_base",
            description="Store verified research text in semantic memory for future retrieval.",
            parameters={
                "type": "object",
                "properties": {
                    "content": {"type": "string"},
                    "title": {"type": "string"},
                    "source": {"type": "string"},
                },
                "required": ["content"],
                "additionalProperties": False,
            },
            handler=lambda content, title="", source="": (
                "Stored knowledge memory "
                + memory.write_knowledge(content, title=title, source=source)
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="ingest_arxiv_paper",
            description=(
                "Download an arXiv paper by ID and save its full extracted text directly to "
                "semantic knowledge memory. Use this instead of copying a truncated tool result."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "string"},
                    "title": {"type": "string"},
                },
                "required": ["arxiv_id"],
                "additionalProperties": False,
            },
            handler=lambda arxiv_id, title="": _ingest_arxiv(
                memory, arxiv_id=arxiv_id, title=title
            ),
        )
    )
    registry.register(
        ToolDefinition(
            name="summarize_and_store",
            description="Compress the active conversation into summary memory and preserve source references.",
            parameters={
                "type": "object",
                "properties": {"thread_id": {"type": "string"}},
                "additionalProperties": False,
            },
            handler=memory.summarize_thread,
        )
    )
    registry.register(
        ToolDefinition(
            name="expand_summary",
            description="Retrieve a stored summary and its source conversation references by summary ID.",
            parameters={
                "type": "object",
                "properties": {"summary_id": {"type": "string"}},
                "required": ["summary_id"],
                "additionalProperties": False,
            },
            handler=memory.expand_summary,
        )
    )


def search_arxiv(query: str, max_results: int = 5) -> str:
    max_results = max(1, min(int(max_results), 10))
    params = urllib.parse.urlencode(
        {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    )
    request = urllib.request.Request(
        f"https://export.arxiv.org/api/query?{params}",
        headers={"User-Agent": "research-memory-agent/0.1"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        root = ET.fromstring(response.read())
    atom = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", atom):
        url = entry.findtext("a:id", default="", namespaces=atom)
        results.append(
            {
                "arxiv_id": url.rsplit("/", 1)[-1],
                "title": " ".join(
                    entry.findtext("a:title", default="", namespaces=atom).split()
                ),
                "authors": [
                    author.findtext("a:name", default="", namespaces=atom)
                    for author in entry.findall("a:author", atom)
                ],
                "abstract": " ".join(
                    entry.findtext("a:summary", default="", namespaces=atom).split()
                ),
                "url": url,
            }
        )
    return json.dumps(results, ensure_ascii=False, indent=2)


def fetch_arxiv_paper(arxiv_id: str, max_characters: int = 50_000) -> str:
    safe_id = arxiv_id.strip().replace("http://arxiv.org/abs/", "").replace(
        "https://arxiv.org/abs/", ""
    )
    if not safe_id or any(not (char.isalnum() or char in ".v/-") for char in safe_id):
        raise ValueError("Invalid arXiv ID")
    request = urllib.request.Request(
        f"https://arxiv.org/pdf/{safe_id}",
        headers={"User-Agent": "research-memory-agent/0.1"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        pdf = response.read()
    reader = PdfReader(io.BytesIO(pdf))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return text[: max(1000, min(int(max_characters), 100_000))]


def _ingest_arxiv(memory: Any, *, arxiv_id: str, title: str = "") -> str:
    text = fetch_arxiv_paper(arxiv_id, max_characters=100_000)
    memory_id = memory.write_knowledge(
        text,
        title=title or f"arXiv:{arxiv_id}",
        source=f"https://arxiv.org/abs/{arxiv_id}",
    )
    return f"Stored {len(text)} characters as semantic memory {memory_id}."
