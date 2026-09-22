from __future__ import annotations

import json
from typing import Any

from .prompts import AGENT_SYSTEM_PROMPT


class ResearchAgent:
    """Bounded read → decide → act → persist research loop."""

    def __init__(
        self,
        *,
        model: Any,
        memory: Any,
        context: Any,
        tools: Any,
        max_iterations: int = 10,
        tool_result_char_limit: int = 3_000,
    ) -> None:
        self.model = model
        self.memory = memory
        self.context = context
        self.tools = tools
        self.max_iterations = max_iterations
        self.tool_result_char_limit = tool_result_char_limit

    def ask(self, query: str, *, thread_id: str = "1") -> str:
        thread_id = str(thread_id)
        bundle = self.context.build(query, thread_id)
        dynamic_tools = self.tools.select(query, limit=5)

        # Persist the request before model/tool work so a mid-run failure does not lose it.
        self.memory.write_message(thread_id, "user", query)
        try:
            self.memory.extract_and_write_entities(query, thread_id)
        except Exception:  # noqa: BLE001, S110 - optional enrichment must not fail the answer
            # Entity extraction is enrichment, not a prerequisite for answering.
            pass

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": bundle.text},
        ]
        steps: list[str] = []
        final_answer = ""

        for iteration in range(1, self.max_iterations + 1):
            message = self.model.complete(messages, tools=dynamic_tools)
            if not getattr(message, "tool_calls", None):
                final_answer = message.content or ""
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )

            for call in message.tool_calls:
                name = call.function.name
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise TypeError("tool arguments must be a JSON object")
                    result = self.tools.execute(name, arguments, thread_id=thread_id)
                    status, error = "success", None
                except Exception as exc:  # noqa: BLE001 - tool boundary records all failures
                    arguments = {}
                    result = f"Error: {exc}"
                    status, error = "failed", str(exc)

                steps.append(f"{name} -> {status}")
                log_id = self.memory.write_tool_log(
                    thread_id=thread_id,
                    tool_call_id=call.id,
                    tool_name=name,
                    tool_args=arguments,
                    result=result,
                    status=status,
                    error_message=error,
                    metadata={"iteration": iteration},
                )
                bounded_result = result
                if len(result) > self.tool_result_char_limit:
                    bounded_result = (
                        result[: self.tool_result_char_limit]
                        + "\n\n[Truncated in model context. Full output is stored as tool log "
                        + log_id
                        + "]"
                    )
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": bounded_result}
                )
        else:
            final_answer = "I could not complete the request within the allowed iterations."

        if steps:
            self.memory.write_workflow(query, steps, final_answer)
        try:
            self.memory.extract_and_write_entities(final_answer, thread_id)
        except Exception:  # noqa: BLE001, S110 - optional enrichment must not fail the answer
            pass
        self.memory.write_message(thread_id, "assistant", final_answer)
        return final_answer
