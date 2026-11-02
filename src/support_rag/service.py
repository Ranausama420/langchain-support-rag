import time
import uuid

from support_rag.graph import build_graph
from support_rag.models import generate
from support_rag.observer import save_run
from support_rag.schemas import Answer, AskRequest, CompareRequest
from support_rag.store import Store


class Service:
    def __init__(self, settings, generator=generate):
        self.settings = settings
        self.store = Store(settings)
        self.generator = generator

    def check_alias(self, alias):
        if alias not in self.settings.models:
            raise ValueError("Unknown model alias; use an alias configured in MODELS_JSON")

    def ask(self, question, model=None, shared_evidence=None):
        request = AskRequest(question=question, model=model)
        alias = model or self.settings.default_model
        self.check_alias(alias)
        start = time.perf_counter()
        graph = build_graph(self.settings, self.store, shared_evidence, self.generator)
        state = graph.invoke(
            {"question": request.question, "model": alias}, config={"run_name": f"support-{alias}"}
        )
        result = Answer(
            run_id=uuid.uuid4().hex,
            model=alias,
            status=state["status"],
            answer=state["answer"],
            claims=state.get("claims", []),
            sources=state.get("evidence", []),
            events=state["events"],
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            usage=state.get("usage", {}),
            error=state.get("error"),
        )
        save_run(self.settings, result, request.question)
        return result

    def compare(self, question, models):
        request = CompareRequest(question=question, models=models)
        for alias in request.models:
            self.check_alias(alias)
        start = time.perf_counter()
        evidence = self.store.retrieve(request.question)
        retrieval_ms = round((time.perf_counter() - start) * 1000, 2)
        results = [self.ask(request.question, alias, evidence) for alias in request.models]
        return {
            "retrieval_ms": retrieval_ms,
            "shared_evidence_ids": [e.id for e in evidence],
            "models": [self.settings.models[a].model_dump() for a in request.models],
            "generation_settings": "Provider defaults; same prompt/schema/evidence for all models",
            "results": [r.model_dump() for r in results],
        }
