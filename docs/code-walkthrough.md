# Code walkthrough

## Configuration and contracts

`config.py` loads `.env` and environment settings using Pydantic. It validates overlap versus chunk size, known provider names, model aliases and live embedding model requirements. Credentials are excluded from object representations. `schemas.py` defines validated API inputs, evidence, source-linked claims and final answers. Comparison requests reject duplicate aliases and allow at most three.

## Embedding and ingestion path

`embeddings.py` implements the LangChain Embeddings interface for the offline simulator and constructs real OpenAI/Ollama clients. Hashing is stable across processes, so offline vectors remain usable after restarting. `ingestion.py` preserves Markdown headings as citation metadata. Content-based identifiers allow unchanged chunks to be updated without duplication.

`store.py` owns Chroma. Explicitly supplied vectors prevent accidental automatic model downloads. Collection names incorporate embedding and chunk configuration. Query results carry distances transformed into bounded cosine similarity. An in-process lock coordinates retrieval with ingestion, but does not provide cross-process transactions.

## Model and graph path

`models.py` constructs LangChain ChatOpenAI, AzureChatOpenAI or ChatOllama clients. A system message treats supplied passages as data, requests evidence-bound claims, and asks the model to abstain when unsupported. Pydantic validates returned JSON. The demo adapter returns a matching passage verbatim and labels its results through its provider configuration.

`graph.py` defines the graph state and public input schema. Clients can submit a question/model alias but cannot supply trusted evidence through the graph input. Node wrappers emit timings and sanitize exceptions. Conditional edges prevent calling generation with no evidence and stop after provider errors. Citation checking verifies ID membership; it intentionally makes no claim of semantic verification.

`service.py` coordinates graph invocation, UUID run IDs, model comparison and persistence. `observer.py` saves redacted summaries unless content storage is explicitly enabled. It validates UUID-shaped identifiers before reading a local run file.

## User-facing entry points

`api.py` exposes local FastAPI endpoints with validated requests and sanitized errors. `ui.py` uses the same service for Ask, Compare and Inspect views. It preserves results in Streamlit session state across reruns. `cli.py` exposes ingestion, answering, comparison and evaluation commands. `studio.py` exports the same graph for graph-level inspection.

## Evaluation

`evaluation.py` runs a fixed synthetic dataset. Expected-term coverage is a deterministic regression signal, not a claim about semantic correctness. Citation validity and abstention are tracked separately. Test fixtures use temporary Chroma directories so tests do not modify the user's index. Provider response tests use fake transports and are not represented as live service tests.
