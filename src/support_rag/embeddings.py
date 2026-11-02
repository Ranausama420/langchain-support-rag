"""Explicit embedding clients; Chroma never silently downloads an embedding model."""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings

STOPWORDS = set(
    "a an the is are was to for of in on and or how do i my it what when can with does you".split()
)


def tokens(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())) - STOPWORDS
    # Small lexical normalization for the offline simulator (configure/configuration).
    # Live providers still receive the original text, without this transformation.
    return {
        re.sub(r"(?:ations|ation|ing|ed|e|s)$", "", word) if len(word) > 5 and "-" not in word else word
        for word in words
    }


class DemoEmbeddings(Embeddings):
    """Deterministic lexical hash vectors for offline plumbing tests, not semantic AI."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        vector = [0.0] * 512
        for token in tokens(text):
            index = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % len(vector)
            vector[index] += 1
        norm = math.sqrt(sum(v * v for v in vector)) or 1
        return [v / norm for v in vector]


def make_embeddings(settings):
    if settings.embedding_provider == "demo":
        return DemoEmbeddings()
    if settings.embedding_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        if not settings.openai_api_key:
            raise ValueError("Set OPENAI_API_KEY to use OpenAI embeddings")
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            request_timeout=settings.model_timeout_seconds,
            max_retries=1,
        )
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.embedding_model,
        base_url=settings.ollama_base_url,
        client_kwargs={"timeout": settings.model_timeout_seconds},
    )
