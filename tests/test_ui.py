from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_ui_ask_flow(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CHROMA_PERSIST_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setenv("RUNS_DIRECTORY", str(tmp_path / "runs"))
    monkeypatch.setenv("MANUALS_DIRECTORY", str(root / "examples/manuals"))
    monkeypatch.setenv("EMBEDDING_PROVIDER", "demo")
    monkeypatch.setenv("MODELS_JSON", '[{"alias":"demo","provider":"demo","model":"extractive-v1"}]')
    monkeypatch.setenv("DEFAULT_MODEL", "demo")
    app = AppTest.from_file(str(root / "ui.py"), default_timeout=30).run()
    assert not app.exception
    app.sidebar.button[0].click().run()
    next(b for b in app.button if b.label == "Ask").click().run()
    assert not app.exception
    assert app.session_state["answer"]["status"] == "answered"
    next(b for b in app.button if b.label == "Compare").click().run()
    assert not app.exception
    assert app.session_state["comparison"]["results"][0]["status"] == "answered"
