import hashlib
import re
from threading import RLock

import chromadb
from chromadb.config import Settings as ChromaSettings

from support_rag.embeddings import make_embeddings
from support_rag.ingestion import load_chunks
from support_rag.schemas import Evidence


class IndexNotReady(ValueError):
    pass


class Store:
    def __init__(self, settings):
        self.settings = settings
        self.lock = RLock()
        self.embeddings = make_embeddings(settings)
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_directory), settings=ChromaSettings(anonymized_telemetry=False)
        )
        # Prevent querying an index with a different embedding space or chunk configuration.
        signature = (
            f"v3:{settings.embedding_provider}:{settings.embedding_model}:"
            f"{settings.chunk_size}:{settings.chunk_overlap}"
        )
        self.name = "manuals-" + hashlib.sha256(signature.encode()).hexdigest()[:16]
        self.collection = self.client.get_or_create_collection(self.name, metadata={"hnsw:space": "cosine"})

    def ingest(self):
        docs = load_chunks(self.settings)
        # Compute all vectors before mutating the stored index.
        vectors = self.embeddings.embed_documents([d.page_content for d in docs])
        ids = [d.metadata["id"] for d in docs]
        with self.lock:
            self.collection.upsert(
                ids=ids,
                embeddings=vectors,
                documents=[d.page_content for d in docs],
                metadatas=[d.metadata for d in docs],
            )
            stale = set(self.collection.get()["ids"]) - set(ids)
            if stale:
                self.collection.delete(ids=sorted(stale))
        return {
            "chunks": len(docs),
            "collection": self.name,
            "embedding_provider": self.settings.embedding_provider,
        }

    def retrieve(self, question):
        codes = set(re.findall(r"\b[A-Z]{2,}-\d+\b", question.upper()))
        filters = [{"$contains": code} for code in sorted(codes)]
        where_document = filters[0] if len(filters) == 1 else {"$and": filters} if filters else None
        with self.lock:
            count = self.collection.count()
            if not count:
                raise IndexNotReady("Index is empty for this configuration. Run support-rag ingest first.")
            result = self.collection.query(
                query_embeddings=[self.embeddings.embed_query(question)],
                n_results=min(count, self.settings.top_k),
                include=["documents", "metadatas", "distances"],
                where_document=where_document,
            )
        items = []
        for cid, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            similarity = max(0.0, min(1.0, 1 - distance))
            if not codes and similarity < self.settings.min_similarity:
                continue
            if codes and not codes.issubset(set(re.findall(r"\b[A-Z]{2,}-\d+\b", text.upper()))):
                continue
            items.append(
                Evidence(
                    id=cid,
                    text=text,
                    source=meta["source"],
                    section=meta["section"],
                    similarity=round(similarity, 5),
                )
            )
        return items
