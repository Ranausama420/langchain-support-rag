# Validation

Validated locally on Windows with Python 3.12.6 on November 2, 2026.

## Completed checks

- Automated suite: **25 passed**. Includes actual persistent Chroma collections, LangGraph routing, source metadata, re-ingestion, configuration isolation, unknown error codes, invalid citations, sanitized provider errors, comparison evidence sharing, FastAPI requests, and Streamlit Ask/Compare interactions.
- Synthetic offline evaluation: **12/12 passed** using demo lexical embeddings and an extractive simulator. Eight answerable and four unanswerable cases. Checks expected terms, source-ID membership and abstention, not semantic correctness of arbitrary answers.
- `ruff check`: passed.
- `pip check`: passed with the direct dependency versions in `requirements.txt`.
- Editable package installation: passed; installed CLI ingestion and evaluation commands executed successfully.
- Live localhost HTTP smoke: FastAPI health, an answered question, an unsupported-code question, and Streamlit health passed.

## Scope of evidence

The offline suite uses real LangGraph execution and Chroma persistence. Live-provider response parsing is exercised with a fake transport. No real OpenAI/Azure credentials or Ollama model server were supplied, so paid model calls, learned embeddings and provider accuracy are **not live-verified**. Docker and authenticated Studio connectivity are provided but not claimed as tested deployments. CI configuration is included; its remote outcome is separate from these local results.

The dataset is a small regression fixture used during implementation, not a held-out benchmark. The lexical simulator normalizes common word suffixes and weights section matches; its score must not be interpreted as LLM quality. Real embeddings and chat models should be evaluated using the same runner plus human assessment on a larger product-specific dataset.

## Reproduce

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
python -m pytest -q
python -m ruff check .
python -m pip check
support-rag ingest
support-rag evaluate --output reports/evaluation.json
```

Use the default no-key demo configuration for these results. Select a configured live model with `support-rag evaluate --model YOUR_ALIAS` to obtain a separate provider report. Regenerate the index when changing embedding settings.
