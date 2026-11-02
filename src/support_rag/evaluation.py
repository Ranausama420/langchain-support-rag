import json


def evaluate(service, dataset, model=None):
    cases = json.loads(dataset.read_text(encoding="utf-8"))
    rows = []
    for case in cases:
        result = service.ask(case["question"], model)
        actual_abstain = result.status == "insufficient_evidence"
        expected_abstain = case["abstain"]
        expected_terms = case.get("expected_terms", [])
        coverage = (
            (sum(term.lower() in result.answer.lower() for term in expected_terms) / len(expected_terms))
            if expected_terms
            else None
        )
        citation_valid = (
            all(set(c.citation_ids).issubset({s.id for s in result.sources}) for c in result.claims)
            if result.claims
            else None
        )
        passed = result.status != "error" and actual_abstain == expected_abstain
        if not expected_abstain:
            passed = passed and coverage == 1 and citation_valid is True
        rows.append(
            {
                "id": case["id"],
                "passed": passed,
                "status": result.status,
                "expected_term_coverage": coverage,
                "citation_ids_valid": citation_valid,
                "latency_ms": result.latency_ms,
                "run_id": result.run_id,
            }
        )
    return {
        "model": model or service.settings.default_model,
        "embedding_provider": service.settings.embedding_provider,
        "method": "Deterministic expected-term coverage and abstention; not semantic entailment or LLM judging",
        "passed": sum(r["passed"] for r in rows),
        "total": len(rows),
        "cases": rows,
    }
