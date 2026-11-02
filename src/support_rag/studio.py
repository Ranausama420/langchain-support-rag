"""Studio entry point. Ingest once before querying this graph."""

from support_rag.config import Settings
from support_rag.graph import build_graph
from support_rag.store import Store

settings = Settings()
graph = build_graph(settings, Store(settings))
