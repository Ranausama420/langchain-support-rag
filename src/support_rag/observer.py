"""Persist safe run summaries; detailed question/source storage is explicitly opt-in."""

import json
import re
from pathlib import Path


def save_run(settings, result, question):
    folder = settings.runs_directory
    folder.mkdir(parents=True, exist_ok=True)
    payload = result.model_dump()
    if settings.store_trace_content:
        payload["question"] = question
    else:
        payload.pop("answer", None)
        payload.pop("claims", None)
        payload["sources"] = [
            {"id": s.id, "source": s.source, "similarity": s.similarity} for s in result.sources
        ]
    path = folder / f"{result.run_id}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_run(settings, run_id):
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("Invalid run identifier")
    return json.loads((Path(settings.runs_directory) / f"{run_id}.json").read_text(encoding="utf-8"))
