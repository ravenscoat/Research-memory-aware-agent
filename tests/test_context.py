from research_memory_agent.context import ContextAssembler, estimate_tokens


class FakeMemory:
    def __init__(self, conversation: str = "short") -> None:
        self.conversation = conversation
        self.summarized = False

    def read_conversation(self, thread_id):
        return "## Conversation Memory\n" + self.conversation

    def read_knowledge(self, query):
        return "## Knowledge Base Memory\nevidence"

    def read_workflow(self, query):
        return "## Workflow Memory\nworkflow"

    def read_entities(self, query):
        return "## Entity Memory\nentity"

    def read_summaries(self, query, thread_id):
        return "## Summary Memory\nsummary"

    def summarize_thread(self, thread_id):
        self.summarized = True
        self.conversation = "compressed"


def test_token_estimate_is_never_zero():
    assert estimate_tokens("") == 1


def test_current_question_is_preserved_after_offload():
    memory = FakeMemory("x" * 2000)
    assembler = ContextAssembler(memory, token_limit=100, offload_threshold=0.5)
    bundle = assembler.build("What is the paper about?", "thread-1")

    assert memory.summarized is True
    assert bundle.offloaded is True
    assert bundle.text.startswith("# Question\nWhat is the paper about?")
    assert "compressed" in bundle.text
