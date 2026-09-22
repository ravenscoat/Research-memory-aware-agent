from research_memory_agent.web import DASHBOARD, app


def test_dashboard_is_agent_chat_without_manual_evidence_ingestion():
    assert app.title == "Memory-Aware Research Agent"
    assert "What would you like to research?" in DASHBOARD
    assert "Ingest evidence" not in DASHBOARD
    assert "PAGE BREAK" not in DASHBOARD
    assert "/api/chat" in DASHBOARD
