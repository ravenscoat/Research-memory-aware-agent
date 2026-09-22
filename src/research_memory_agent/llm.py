from __future__ import annotations

import json
import uuid
from types import SimpleNamespace
from typing import Any
from urllib.request import Request, urlopen


class OllamaChatModel:
    """Small Ollama adapter with the interface used by the research agent."""

    def __init__(self, *, base_url: str, model: str, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Any:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._ollama_messages(messages),
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1},
        }
        if tools:
            payload["tools"] = tools
        raw = self._post("/api/chat", payload).get("message", {})
        calls = []
        for call in raw.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments") or {}
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            calls.append(
                SimpleNamespace(
                    id=str(call.get("id") or uuid.uuid4()),
                    function=SimpleNamespace(
                        name=str(function.get("name") or ""), arguments=arguments
                    ),
                )
            )
        return SimpleNamespace(content=raw.get("content") or "", tool_calls=calls)

    @staticmethod
    def _ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style string tool arguments back to Ollama JSON objects."""
        normalized: list[dict[str, Any]] = []
        for source in messages:
            message = dict(source)
            if source.get("tool_calls"):
                converted = []
                for source_call in source["tool_calls"]:
                    call = dict(source_call)
                    function = dict(call.get("function") or {})
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            function["arguments"] = json.loads(arguments)
                        except json.JSONDecodeError:
                            function["arguments"] = {}
                    call["function"] = function
                    converted.append(call)
                message["tool_calls"] = converted
            normalized.append(message)
        return normalized

    def _text(self, system: str, user: str, *, json_format: bool = False) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.0},
        }
        if json_format:
            payload["format"] = "json"
        return str(self._post("/api/chat", payload).get("message", {}).get("content") or "")

    def summarize(self, transcript: str) -> str:
        return self._text(
            "Summarize faithfully. Preserve decisions, facts, open questions, source names, "
            "and user preferences. Do not invent information.",
            transcript,
        )

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        raw = self._text(
            "Extract important named entities. Return JSON only as "
            '{"entities":[{"name":"...","type":"...","description":"..."}]}.',
            text,
            json_format=True,
        )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return []
        entities = value.get("entities", []) if isinstance(value, dict) else []
        return [item for item in entities if isinstance(item, dict)]

    def health(self) -> bool:
        try:
            with urlopen(self.base_url + "/api/tags", timeout=5) as response:
                return response.status == 200
        except OSError:
            return False
