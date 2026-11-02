"""Graph nodes are visible in Studio; public input cannot inject retrieved passages."""

import time
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from support_rag.models import generate
from support_rag.schemas import Draft, Evidence

ABSTENTION = "The available manuals do not provide enough evidence to answer this question."


class InputState(TypedDict):
    question: str
    model: str


class State(InputState, total=False):
    evidence: list[dict]
    sufficient: bool
    draft: dict
    usage: dict
    status: str
    answer: str
    claims: list[dict]
    error: str
    events: list[dict]


def build_graph(settings, store, shared_evidence=None, generator=generate):
    def observe(name, fn):
        def wrapped(state):
            start = time.perf_counter()
            events = list(state.get("events", []))
            events.append({"node": name, "event": "started"})
            try:
                update = fn(state)
                events.append(
                    {
                        "node": name,
                        "event": "completed",
                        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                )
            except Exception:
                # Provider exceptions can embed URLs, tokens or prompt text. Never expose them.
                update = {
                    "status": "error",
                    "error": f"{name} failed; check configuration and provider availability",
                    "answer": "Unable to complete this request.",
                    "claims": [],
                }
                events.append(
                    {
                        "node": name,
                        "event": "error",
                        "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                    }
                )
            update["events"] = events
            return update

        return wrapped

    def retrieve(state):
        items = shared_evidence if shared_evidence is not None else store.retrieve(state["question"])
        return {"evidence": [e.model_dump() for e in items]}

    def assess(state):
        return {"sufficient": bool(state["evidence"])}

    def answer(state):
        spec = settings.models[state.get("model") or settings.default_model]
        draft, usage = generator(
            spec, settings, state["question"], [Evidence.model_validate(e) for e in state["evidence"]]
        )
        return {"draft": draft.model_dump(), "usage": usage}

    def abstain(state):
        return {"status": "insufficient_evidence", "answer": ABSTENTION, "claims": []}

    def validate(state):
        draft = Draft.model_validate(state["draft"])
        if draft.insufficient_evidence or not draft.claims:
            return abstain(state)
        allowed = {e["id"] for e in state["evidence"]}
        if any(not set(c.citation_ids).issubset(allowed) for c in draft.claims):
            return abstain(state)
        # ID validity is not semantic entailment; evaluation measures coverage separately.
        return {
            "status": "answered",
            "claims": [c.model_dump() for c in draft.claims],
            "answer": "\n\n".join(
                c.text + " " + " ".join(f"[{i}]" for i in c.citation_ids) for c in draft.claims
            ),
        }

    graph = StateGraph(State, input_schema=InputState)
    for name, fn in [
        ("retrieve", retrieve),
        ("assess_evidence", assess),
        ("generate", answer),
        ("validate_citations", validate),
        ("abstain", abstain),
    ]:
        graph.add_node(name, observe(name, fn))
    graph.add_edge(START, "retrieve")
    graph.add_conditional_edges(
        "retrieve", lambda s: "error" if s.get("error") else "ok", {"error": END, "ok": "assess_evidence"}
    )
    graph.add_conditional_edges(
        "assess_evidence",
        lambda s: "yes" if s.get("sufficient") else "no",
        {"yes": "generate", "no": "abstain"},
    )
    graph.add_conditional_edges(
        "generate", lambda s: "error" if s.get("error") else "ok", {"error": END, "ok": "validate_citations"}
    )
    graph.add_edge("validate_citations", END)
    graph.add_edge("abstain", END)
    return graph.compile()
