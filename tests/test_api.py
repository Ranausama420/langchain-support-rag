from fastapi.testclient import TestClient

from support_rag import api


def test_api_end_to_end(service, monkeypatch):
    monkeypatch.setattr(api, "get_service", lambda: service)
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/ingest").json()["chunks"] > 0
        answer = client.post("/ask", json={"question": "How do I fix AUTH-104?"}).json()
        assert answer["status"] == "answered"
        assert client.get("/runs/" + answer["run_id"]).status_code == 200
        assert client.post("/ask", json={"question": "  "}).status_code == 422
        assert client.post("/ask", json={"question": "test", "model": "unknown"}).status_code == 400
        assert client.get("/runs/not-a-run").status_code == 404
        comparison = client.post("/compare", json={"question": "How do I fix AUTH-104?", "models": ["demo"]})
        assert comparison.json()["results"][0]["status"] == "answered"
        assert (
            client.post("/compare", json={"question": "test", "models": ["demo", "demo"]}).status_code == 422
        )
