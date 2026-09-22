from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class MemoryType(StrEnum):
    CONVERSATION = "conversation"
    SEMANTIC = "semantic"
    WORKFLOW = "workflow"
    TOOLBOX = "toolbox"
    ENTITY = "entity"
    SUMMARY = "summary"
    TOOL_LOG = "tool_log"


@dataclass(frozen=True)
class MemoryItem:
    id: str
    memory_type: MemoryType
    text: str
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., str]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
