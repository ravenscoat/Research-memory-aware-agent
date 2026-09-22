from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def estimate_tokens(text: str) -> int:
    """Conservative provider-independent estimate used for context budgeting."""
    return max(1, (len(text) + 3) // 4)


@dataclass(frozen=True)
class ContextBundle:
    text: str
    estimated_tokens: int
    usage_ratio: float
    offloaded: bool


class ContextAssembler:
    def __init__(self, memory: Any, *, token_limit: int, offload_threshold: float) -> None:
        self.memory = memory
        self.token_limit = token_limit
        self.offload_threshold = offload_threshold

    def build(self, query: str, thread_id: str) -> ContextBundle:
        memory_text = self._memory_text(query, thread_id)
        tokens = estimate_tokens(memory_text)
        ratio = tokens / self.token_limit
        offloaded = False

        if ratio > self.offload_threshold:
            self.memory.summarize_thread(thread_id)
            # Put the newly created summary before lower-priority recalled memories.
            memory_text = self._memory_text(query, thread_id, compact=True)
            tokens = estimate_tokens(memory_text)
            memory_budget = max(1, int(self.token_limit * self.offload_threshold))
            if tokens > memory_budget:
                # Four characters per token matches estimate_tokens. The current question is
                # appended afterwards, so it can never be cut by this fallback bound.
                memory_text = memory_text[: memory_budget * 4]
                tokens = estimate_tokens(memory_text)
            ratio = tokens / self.token_limit
            offloaded = True

        # The live question is added only after compaction, so it is never summarized away.
        text = f"# Question\n{query}\n\n{memory_text}"
        return ContextBundle(
            text=text,
            estimated_tokens=estimate_tokens(text),
            usage_ratio=ratio,
            offloaded=offloaded,
        )

    def _memory_text(self, query: str, thread_id: str, *, compact: bool = False) -> str:
        conversation = self.memory.read_conversation(thread_id)
        knowledge = self.memory.read_knowledge(query)
        workflow = self.memory.read_workflow(query)
        entities = self.memory.read_entities(query)
        summaries = self.memory.read_summaries(query, thread_id)
        sections = (
            [conversation, summaries, knowledge, entities, workflow]
            if compact
            else [conversation, knowledge, workflow, entities, summaries]
        )
        return "\n\n".join(sections)
