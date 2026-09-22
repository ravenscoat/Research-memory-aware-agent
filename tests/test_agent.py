from dataclasses import dataclass

from research_memory_agent.agent import ResearchAgent
from research_memory_agent.context import ContextBundle


@dataclass
class Message:
    content: str
    tool_calls: list | None = None


class FakeModel:
    def complete(self, messages, tools=None):
        return Message("A grounded answer.")


class FakeContext:
    def build(self, query, thread_id):
        return ContextBundle(query, 5, 0.01, False)


class FakeTools:
    def select(self, query, limit=5):
        return []


class FakeMemory:
    def __init__(self):
        self.messages = []

    def write_message(self, thread_id, role, content):
        self.messages.append((thread_id, role, content))

    def extract_and_write_entities(self, text, thread_id=None):
        return []

    def write_workflow(self, query, steps, answer):
        raise AssertionError("No workflow should be stored when no tool ran")


def test_plain_answer_is_written_to_conversation():
    memory = FakeMemory()
    agent = ResearchAgent(
        model=FakeModel(),
        memory=memory,
        context=FakeContext(),
        tools=FakeTools(),
    )

    answer = agent.ask("question", thread_id="abc")

    assert answer == "A grounded answer."
    assert memory.messages == [
        ("abc", "user", "question"),
        ("abc", "assistant", "A grounded answer."),
    ]
