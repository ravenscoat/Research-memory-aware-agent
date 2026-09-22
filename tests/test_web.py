from research_memory_agent.web import DASHBOARD, app


def test_dashboard_exposes_evidence_and_verification_traces():
    assert app.title == "Evidence Research Workspace"
    assert "Retrieved memory" in DASHBOARD
    assert "Claim verification" in DASHBOARD
    assert "split('\\nPAGE BREAK\\n')" in DASHBOARD
