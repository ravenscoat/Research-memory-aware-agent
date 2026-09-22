from __future__ import annotations

import json
from typing import Any

from .prompts import ENTITY_EXTRACTION_PROMPT


class OpenAIChatModel:
    """Small adapter around OpenAI Chat Completions used by the agent and memory layer."""

    def __init__(self, model: str, client: Any | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model

    def complete(self, messages: list[dict[str, Any]], tools: list[dict] | None = None) -> Any:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs.update(tools=tools, tool_choice="auto")
        return self.client.chat.completions.create(**kwargs).choices[0].message

    def summarize(self, text: str) -> str:
        message = self.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "Compress the conversation into a factual durable summary. Preserve user "
                        "goals, referenced papers, decisions, unresolved work, and identifiers."
                    ),
                },
                {"role": "user", "content": text},
            ]
        )
        return message.content or ""

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": ENTITY_EXTRACTION_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        try:
            items = json.loads(raw).get("entities", [])
        except json.JSONDecodeError:
            return []
        return [item for item in items if isinstance(item, dict) and item.get("name")]
