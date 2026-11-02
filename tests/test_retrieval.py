from support_rag.ingestion import load_chunks
from support_rag.store import Store


def test_persistence_and_source_metadata(service):
    reopened = Store(service.settings)
    evidence = reopened.retrieve("How do I fix AUTH-104?")
    assert evidence and evidence[0].source == "authentication.md"
    assert "AUTH-104" in evidence[0].section
    stored = reopened.collection.get(include=["embeddings"])
    assert len(stored["embeddings"][0]) == 512


def test_unknown_error_code_never_matches_other_codes(service):
    assert service.store.retrieve("How do I fix AUTH-999?") == []


def test_reingestion_removes_stale_chunks(settings, tmp_path):
    manuals = tmp_path / "manuals"
    manuals.mkdir()
    path = manuals / "policy.md"
    path.write_text("# Policy\n## Retention\nRetention is 14 days.")
    settings.manuals_directory = manuals
    store = Store(settings)
    store.ingest()
    old_ids = store.collection.get()["ids"]
    path.write_text("# Policy\n## Retention\nRetention is 30 days.")
    store.ingest()
    assert not set(old_ids) & set(store.collection.get()["ids"])
    assert store.collection.count() == 1


def test_chunk_bounds(settings):
    docs = load_chunks(settings)
    assert all(len(d.page_content) <= settings.chunk_size for d in docs)
    assert all(d.metadata["source"] and d.metadata["section"] for d in docs)


def test_embedding_config_isolated(service):
    changed = service.settings.model_copy(update={"embedding_model": "different-signature"})
    store = Store(changed)
    assert store.name != service.store.name
    assert store.collection.count() == 0
