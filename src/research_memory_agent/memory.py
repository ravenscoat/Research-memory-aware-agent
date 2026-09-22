from __future__ import annotations

import json
from typing import Any

from .models import MemoryType


class MemoryManager:
    """Application-level read/write API across all seven memory categories."""

    def __init__(self, store: Any, llm: Any) -> None:
        self.store = store
        self.llm = llm

    def read_conversation(self, thread_id: str, *, limit: int = 12) -> str:
        rows = self.store.recent_conversation(thread_id, limit=limit)
        body = "\n".join(f"{row['role']}: {row['content']}" for row in rows)
        return _section("Conversation Memory", body)

    def read_knowledge(self, query: str, *, limit: int = 4) -> str:
        return _items_section(
            "Knowledge Base Memory",
            self.store.search_vector(MemoryType.SEMANTIC, query, limit=limit),
        )

    def read_workflow(self, query: str, *, limit: int = 3) -> str:
        return _items_section(
            "Workflow Memory",
            self.store.search_vector(MemoryType.WORKFLOW, query, limit=limit),
        )

    def read_entities(self, query: str, *, limit: int = 4) -> str:
        return _items_section(
            "Entity Memory",
            self.store.search_vector(MemoryType.ENTITY, query, limit=limit),
        )

    def read_summaries(self, query: str, thread_id: str, *, limit: int = 3) -> str:
        return _items_section(
            "Summary Memory",
            self.store.search_vector(
                MemoryType.SUMMARY, query, limit=limit, thread_id=thread_id
            ),
            show_ids=True,
        )

    def select_tools(self, query: str, *, limit: int = 5) -> list[str]:
        records = self.store.search_vector(MemoryType.TOOLBOX, query, limit=limit)
        return [str(record.metadata.get("name")) for record in records if record.metadata.get("name")]

    def write_message(self, thread_id: str, role: str, content: str) -> str:
        return self.store.append_conversation(thread_id, role, content)

    def write_knowledge(
        self, text: str, *, title: str = "", source: str = "", thread_id: str | None = None
    ) -> str:
        return self.store.append_vector(
            MemoryType.SEMANTIC,
            text,
            thread_id=thread_id,
            metadata={"title": title, "source": source},
        )

    def write_workflow(self, query: str, steps: list[str], answer: str) -> str:
        text = f"Goal: {query}\nSteps:\n" + "\n".join(f"- {step}" for step in steps)
        text += f"\nOutcome: {answer[:1500]}"
        return self.store.append_vector(
            MemoryType.WORKFLOW,
            text,
            metadata={"step_count": len(steps)},
        )

    def extract_and_write_entities(self, text: str, thread_id: str | None = None) -> list[str]:
        ids: list[str] = []
        for entity in self.llm.extract_entities(text):
            name = str(entity.get("name", "")).strip()
            if not name:
                continue
            entity_type = str(entity.get("type", "unknown"))
            description = str(entity.get("description", ""))
            ids.append(
                self.store.append_vector(
                    MemoryType.ENTITY,
                    f"{name} ({entity_type}): {description}",
                    thread_id=thread_id,
                    metadata={"name": name, "type": entity_type},
                )
            )
        return ids

    def summarize_thread(self, thread_id: str) -> str:
        rows = self.store.recent_conversation(
            thread_id, limit=100, include_summarized=False
        )
        if not rows:
            return "No unsummarized conversation was available."
        transcript = "\n".join(f"{row['role']}: {row['content']}" for row in rows)
        summary = self.llm.summarize(transcript)
        source_ids = [row["id"] for row in rows]
        summary_id = self.store.append_vector(
            MemoryType.SUMMARY,
            summary,
            thread_id=thread_id,
            metadata={"source_count": len(source_ids)},
            source_ids=source_ids,
        )
        self.store.mark_conversation_summarized(thread_id, source_ids)
        return f"Stored summary {summary_id}: {summary}"

    def expand_summary(self, summary_id: str) -> str:
        item = self.store.get_vector(MemoryType.SUMMARY, summary_id)
        if item is None:
            return f"Summary {summary_id} was not found."
        source_messages = self.store.conversation_by_ids(
            item.thread_id, item.source_ids
        ) if item.thread_id else []
        return json.dumps(
            {
                "summary_id": item.id,
                "summary": item.text,
                "source_conversation": source_messages,
            },
            ensure_ascii=False,
            indent=2,
        )

    def write_tool_log(self, **kwargs: Any) -> str:
        return self.store.write_tool_log(**kwargs)


def _section(name: str, body: str) -> str:
    return f"## {name}\n{body or '[No relevant memory]'}"


def _items_section(name: str, items: list[Any], *, show_ids: bool = False) -> str:
    lines = []
    for item in items:
        prefix = f"[{item.id}] " if show_ids else ""
        lines.append(prefix + item.text)
    return _section(name, "\n\n".join(lines))
