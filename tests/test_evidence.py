from types import SimpleNamespace

from research_memory_agent.evidence import EvidenceGroundedResearch, chunk_page
from research_memory_agent.models import MemoryItem, MemoryType


class FakeStore:
    def __init__(self):
        self.writes = []
        self.trace = None

    def append_vector(self, memory_type, text, metadata):
        self.writes.append((memory_type, text, metadata))
        return f"chunk-{len(self.writes)}"

    def deactivate_document_chunks(self, document_id):
        return 0

    def search_hybrid_evidence(self, question, limit=6):
        return [
            MemoryItem(
                id="chunk-1",
                memory_type=MemoryType.SEMANTIC,
                text="Hybrid retrieval combines vector similarity and keyword search.",
                metadata={
                    "kind": "evidence_chunk", "document_id": "doc-1", "title": "Guide",
                    "source": "local", "page_number": 2, "chunk_index": 0,
                },
                score=0.9,
            )
        ]

    def write_evidence_trace(self, **kwargs):
        self.trace = kwargs
        return "trace-1"


class FakeModel:
    def complete(self, messages):
        return SimpleNamespace(
            content="Hybrid retrieval combines vector similarity and keyword search. [Guide, p. 2, chunk 0]"
        )


def test_page_chunking_preserves_page_metadata():
    store = FakeStore()
    workflow = EvidenceGroundedResearch(store=store, model=FakeModel())

    result = workflow.ingest_pages(["one " * 700, "two " * 20], title="Paper", source="test")

    assert result["pages"] == 2
    assert result["chunks"] >= 3
    assert {item[2]["page_number"] for item in store.writes} == {1, 2}
    assert all(item[2]["document_id"] == result["document_id"] for item in store.writes)


def test_grounded_answer_is_verified_and_traced():
    store = FakeStore()
    workflow = EvidenceGroundedResearch(store=store, model=FakeModel())

    result = workflow.answer("What does hybrid retrieval combine?")

    assert result["trace_id"] == "trace-1"
    assert result["all_claims_supported"] is True
    assert store.trace is not None


def test_chunk_page_uses_overlap_without_empty_chunks():
    chunks = chunk_page("word " * 600, size=200, overlap=40)

    assert len(chunks) > 2
    assert all(chunks)
