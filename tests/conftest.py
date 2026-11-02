from pathlib import Path

import pytest

from support_rag.config import Settings
from support_rag.service import Service

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        embedding_provider="demo",
        embedding_model="",
        models_json='[{"alias":"demo","provider":"demo","model":"extractive-v1"}]',
        default_model="demo",
        chroma_persist_directory=tmp_path / "chroma",
        runs_directory=tmp_path / "runs",
        manuals_directory=ROOT / "examples/manuals",
        store_trace_content=False,
    )


@pytest.fixture
def service(settings):
    result = Service(settings)
    result.store.ingest()
    return result
