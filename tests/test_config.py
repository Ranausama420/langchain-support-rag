import json

import pytest
from pydantic import ValidationError

from support_rag.config import Settings
from support_rag.schemas import CompareRequest


def test_env_file_loading(tmp_path, monkeypatch):
    for name in ["TOP_K", "DEFAULT_MODEL", "MODELS_JSON"]:
        monkeypatch.delenv(name, raising=False)
    env = tmp_path / ".env"
    env.write_text(
        'TOP_K=2\nMODELS_JSON=[{"alias":"fixture","provider":"demo","model":"v1"}]\nDEFAULT_MODEL=fixture\n',
        encoding="utf-8",
    )
    settings = Settings(_env_file=env)
    assert settings.top_k == 2 and settings.default_model == "fixture"
    monkeypatch.setenv("TOP_K", "3")
    assert Settings(_env_file=env).top_k == 3


def test_invalid_chunk_configuration():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, chunk_size=100, chunk_overlap=101)


def test_duplicate_model_aliases():
    specs = [{"alias": "demo", "provider": "demo", "model": "a"}] * 2
    with pytest.raises(ValidationError):
        Settings(_env_file=None, models_json=json.dumps(specs))


def test_comparison_limit():
    with pytest.raises(ValidationError):
        CompareRequest(question="test question", models=["a", "b", "c", "d"])


def test_secret_not_in_settings_repr():
    settings = Settings(_env_file=None, openai_api_key="test-secret-never-a-real-key")
    assert "test-secret-never-a-real-key" not in repr(settings)
