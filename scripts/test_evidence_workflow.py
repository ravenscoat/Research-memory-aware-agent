"""Live end-to-end evidence workflow check using a complete arXiv paper."""

import os

from research_memory_agent.app import build_application

paper_id = os.getenv("EVIDENCE_TEST_ARXIV_ID", "1706.03762")
app = build_application()
try:
    ingested = app.evidence.ingest_arxiv(paper_id, title="Attention Is All You Need")
    result = app.evidence.answer(
        "Why does the Transformer use multi-head attention?",
        thread_id="full-paper-evidence-test",
    )
    assert ingested["pages"] > 1
    assert ingested["chunks"] >= ingested["pages"]
    assert result["evidence"]
    assert result["verification"]
    print("DOCUMENT_ID=" + ingested["document_id"])
    print("PAGES=" + str(ingested["pages"]))
    print("CHUNKS=" + str(ingested["chunks"]))
    print("ANSWER=" + result["answer"].replace("\n", " "))
    print("EVIDENCE_COUNT=" + str(len(result["evidence"])))
    print("CLAIMS_SUPPORTED=" + str(result["all_claims_supported"]))
    print("TRACE_ID=" + result["trace_id"])
finally:
    app.close()
