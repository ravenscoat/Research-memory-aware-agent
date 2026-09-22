from dataclasses import dataclass
from types import SimpleNamespace

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


class RepeatingToolModel:
    def complete(self, messages, tools=None):
        call = SimpleNamespace(
            id="repeat-call",
            function=SimpleNamespace(name="test_tool", arguments='{"value":1}'),
        )
        return Message("", [call])


class LoopMemory(FakeMemory):
    def __init__(self):
        super().__init__()
        self.logs = []
        self.workflow = None

    def write_tool_log(self, **kwargs):
        self.logs.append(kwargs)
        return f"log-{len(self.logs)}"

    def write_workflow(self, query, steps, answer):
        self.workflow = (query, steps, answer)


class LoopTools(FakeTools):
    def execute(self, name, arguments, thread_id):
        return "tool result"


def test_repeating_tool_calls_stop_at_iteration_limit():
    memory = LoopMemory()
    agent = ResearchAgent(
        model=RepeatingToolModel(),
        memory=memory,
        context=FakeContext(),
        tools=LoopTools(),
        max_iterations=2,
    )

    answer = agent.ask("loop forever", thread_id="bounded")

    assert answer == "I could not complete the request within the allowed iterations."
    assert len(memory.logs) == 2
    assert memory.workflow is not None
