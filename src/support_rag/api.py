from functools import lru_cache

from fastapi import FastAPI, HTTPException

from support_rag.config import Settings
from support_rag.observer import read_run
from support_rag.schemas import AskRequest, CompareRequest
from support_rag.service import Service


@lru_cache
def get_service():
    return Service(Settings())


app = FastAPI(title="NimbusDesk Support RAG", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def models():
    return [m.model_dump() for m in get_service().settings.models.values()]


@app.post("/ingest")
def ingest():
    try:
        return get_service().store.ingest()
    except ValueError:
        raise HTTPException(400, "Check manual files and embedding configuration") from None
    except Exception:
        raise HTTPException(503, "Ingestion unavailable; check configuration") from None


@app.post("/ask")
def ask(request: AskRequest):
    try:
        return get_service().ask(request.question, request.model)
    except ValueError:
        raise HTTPException(400, "Check the question and configured model alias") from None
    except Exception:
        raise HTTPException(503, "Service unavailable; check configuration") from None


@app.post("/compare")
def compare(request: CompareRequest):
    try:
        return get_service().compare(request.question, request.models)
    except ValueError:
        raise HTTPException(400, "Check model aliases and ingest documents first") from None
    except Exception:
        raise HTTPException(503, "Comparison unavailable; check configuration") from None


@app.get("/runs/{run_id}")
def run(run_id: str):
    try:
        return read_run(get_service().settings, run_id)
    except (ValueError, FileNotFoundError):
        raise HTTPException(404, "Run not found") from None
