import json

from support_rag.graph import build_graph
from support_rag.observer import read_run
from support_rag.schemas import Draft
from support_rag.service import Service


def test_answer_and_observer(service):
    result = service.ask("How do I fix AUTH-104?")
    assert result.status == "answered"
    assert "expired" in result.answer
    assert result.claims[0].citation_ids[0] in {s.id for s in result.sources}
    completed = [e["node"] for e in result.events if e["event"] == "completed"]
    assert completed == ["retrieve", "assess_evidence", "generate", "validate_citations"]


def test_no_evidence_bypasses_model(service):
    def forbidden(*args):
        raise AssertionError("Generation should not be called")

    service.generator = forbidden
    result = service.ask("How do I fix AUTH-999?")
    assert result.status == "insufficient_evidence"
    assert "generate" not in [e["node"] for e in result.events]


def test_fabricated_citation_rejected(service):
    service.generator = lambda *args: (
        Draft(
            insufficient_evidence=False, claims=[{"text": "Invented fact", "citation_ids": ["nonexistent"]}]
        ),
        {},
    )
    result = service.ask("How do I fix AUTH-104?")
    assert result.status == "insufficient_evidence"
    assert not result.claims


def test_provider_failure_redacted(service):
    def broken(*args):
        raise RuntimeError("secret-token-must-not-leak")

    service.generator = broken
    result = service.ask("How do I fix AUTH-104?")
    assert result.status == "error"
    assert "secret-token" not in result.model_dump_json()


def test_default_disk_trace_has_no_content(service):
    result = service.ask("How do I fix AUTH-104?")
    stored = read_run(service.settings, result.run_id)
    assert "question" not in stored and "answer" not in stored and "claims" not in stored
    assert all("text" not in s for s in stored["sources"])


def test_comparison_retrieves_once_and_isolates_errors(settings, monkeypatch):
    settings.models_json = json.dumps(
        [{"alias": a, "provider": "demo", "model": "extractive-v1"} for a in ["good", "bad"]]
    )
    settings.default_model = "good"
    from support_rag.models import generate

    seen = []

    def generator(spec, config, question, evidence):
        seen.append([e.model_dump() for e in evidence])
        if spec.alias == "bad":
            raise RuntimeError("provider unavailable")
        return generate(spec, config, question, evidence)

    service = Service(settings, generator)
    service.store.ingest()
    calls = []
    original = service.store.retrieve

    def counted(q):
        calls.append(q)
        return original(q)

    monkeypatch.setattr(service.store, "retrieve", counted)
    result = service.compare("How do I fix AUTH-104?", ["good", "bad"])
    assert len(calls) == 1 and seen[0] == seen[1]
    assert [r["status"] for r in result["results"]] == ["answered", "error"]


def test_graph_input_cannot_supply_evidence(service):
    graph = build_graph(service.settings, service.store)
    result = graph.invoke(
        {
            "question": "How do I fix AUTH-999?",
            "model": "demo",
            "evidence": [{"id": "fake", "text": "fabricated"}],
        }
    )
    assert result["status"] == "insufficient_evidence"


def test_empty_index_returns_actionable_error(settings):
    result = Service(settings).ask("How do I fix AUTH-104?")
    assert result.status == "error"
