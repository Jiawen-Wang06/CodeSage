import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import chromadb
from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from codesage.ingest import remove_tree

COLLECTION_NAME = "code_chunks"
MANIFEST_FILE = "manifest.json"

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_SUBWORD = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

EmbedProgress = Callable[[int, int], None]


@dataclass(frozen=True)
class RetrievalResult:
    """Everything the hybrid retriever did for one query, for tracing and evaluation."""

    query: str
    query_tokens: list[str]
    dense: list[Document]
    sparse: list[tuple[Document, float]]
    fused: list[tuple[Document, float]]
    documents: list[Document]
    timings_ms: dict[str, float]


class _UsageRegistry:
    def __init__(self):
        self._counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def acquire(self, key: str) -> None:
        with self._lock:
            self._counts[key] += 1

    def release(self, key: str) -> None:
        with self._lock:
            self._counts[key] -= 1
            if self._counts[key] <= 0:
                del self._counts[key]

    def in_use(self, key: str) -> bool:
        with self._lock:
            return self._counts[key] > 0


_USAGE = _UsageRegistry()


@dataclass
class RepoIndex:
    client: chromadb.ClientAPI
    vector_store: Chroma
    documents: list[Document]
    usage_key: str
    closed: bool = False

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.client.close()
        _USAGE.release(self.usage_key)


@dataclass(frozen=True)
class StoredChunk:
    document: Document
    embedding: list[float]


class IndexStore:
    def __init__(self, root: Path, embeddings: Embeddings, batch_size: int, fingerprint: dict):
        self._root = root
        self._embeddings = embeddings
        self._batch_size = batch_size
        self._fingerprint = fingerprint

    def exists(self, name: str) -> bool:
        manifest = self._read_manifest(name)
        return manifest is not None and manifest.get("fingerprint") == self._fingerprint

    def load(self, name: str) -> RepoIndex:
        client, vector_store = self._open(name)
        stored = vector_store.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=content, metadata=metadata or {})
            for content, metadata in zip(stored["documents"], stored["metadatas"])
        ]
        return self._track(name, client, vector_store, documents)

    def latest(self, prefix: str, exclude: str) -> str | None:
        candidates = [
            (manifest["created_at"], path.name)
            for path in self._versions(prefix)
            if path.name != exclude
            and (manifest := self._read_manifest(path.name)) is not None
            and manifest.get("fingerprint") == self._fingerprint
        ]
        return max(candidates)[1] if candidates else None

    def stored_chunks(self, name: str) -> dict[tuple[str, str], list[StoredChunk]]:
        client, _ = self._open(name)
        try:
            stored = client.get_collection(COLLECTION_NAME).get(include=["documents", "metadatas", "embeddings"])
        finally:
            client.close()
        chunks: dict[tuple[str, str], list[StoredChunk]] = {}
        for content, metadata, embedding in zip(stored["documents"], stored["metadatas"], stored["embeddings"]):
            chunk = StoredChunk(Document(page_content=content, metadata=metadata), [float(x) for x in embedding])
            chunks.setdefault((metadata["path"], metadata["file_hash"]), []).append(chunk)
        return chunks

    def build(
        self,
        name: str,
        new_documents: list[Document],
        reused: list[StoredChunk],
        on_progress: EmbedProgress | None = None,
    ) -> RepoIndex:
        remove_tree(self._root / name)
        client, vector_store = self._open(name)
        self._add_reused(client, reused)

        total = len(new_documents)
        for start in range(0, total, self._batch_size):
            batch = new_documents[start:start + self._batch_size]
            vector_store.add_documents(batch, ids=[f"n{start + i}" for i in range(len(batch))])
            if on_progress:
                on_progress(start + len(batch), total)

        documents = [chunk.document for chunk in reused] + new_documents
        self._write_manifest(name, len(documents))
        return self._track(name, client, vector_store, documents)

    def prune(self, prefix: str, keep: str) -> None:
        for path in self._versions(prefix):
            if path.name == keep or _USAGE.in_use(self._usage_key(path.name)):
                continue
            (path / MANIFEST_FILE).unlink(missing_ok=True)
            try:
                remove_tree(path)
            except OSError:
                pass

    def _track(
        self, name: str, client: chromadb.ClientAPI, vector_store: Chroma, documents: list[Document]
    ) -> RepoIndex:
        usage_key = self._usage_key(name)
        _USAGE.acquire(usage_key)
        return RepoIndex(client=client, vector_store=vector_store, documents=documents, usage_key=usage_key)

    def _usage_key(self, name: str) -> str:
        return str((self._root / name).resolve())

    def _versions(self, prefix: str) -> list[Path]:
        if not self._root.is_dir():
            return []
        return [path for path in self._root.iterdir() if path.is_dir() and path.name.startswith(f"{prefix}_")]

    def _add_reused(self, client: chromadb.ClientAPI, reused: list[StoredChunk]) -> None:
        collection = client.get_collection(COLLECTION_NAME)
        max_batch = client.get_max_batch_size()
        for start in range(0, len(reused), max_batch):
            batch = reused[start:start + max_batch]
            collection.add(
                ids=[f"r{start + i}" for i in range(len(batch))],
                embeddings=[chunk.embedding for chunk in batch],
                documents=[chunk.document.page_content for chunk in batch],
                metadatas=[chunk.document.metadata for chunk in batch],
            )

    def _open(self, name: str) -> tuple[chromadb.ClientAPI, Chroma]:
        path = self._root / name
        path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(
            path=str(path),
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
        vector_store = Chroma(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding_function=self._embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        return client, vector_store

    def _read_manifest(self, name: str) -> dict | None:
        path = self._root / name / MANIFEST_FILE
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write_manifest(self, name: str, chunk_count: int) -> None:
        manifest = {
            "name": name,
            "chunk_count": chunk_count,
            "fingerprint": self._fingerprint,
            "created_at": time.time(),
        }
        (self._root / name / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")


class HybridRetriever(BaseRetriever):
    vector_store: Chroma
    embeddings: Embeddings
    documents: list[Document]
    bm25: BM25Okapi
    k: int = 8
    candidate_k: int = 20
    rrf_k: int = 60

    @classmethod
    def from_index(cls, index: RepoIndex, embeddings: Embeddings, k: int) -> "HybridRetriever":
        corpus = [tokenize(doc.page_content) for doc in index.documents]
        return cls(
            vector_store=index.vector_store,
            embeddings=embeddings,
            documents=index.documents,
            bm25=BM25Okapi(corpus),
            k=k,
            candidate_k=max(k * 3, 20),
        )

    def with_embeddings(self, embeddings: Embeddings) -> "HybridRetriever":
        return self.model_copy(update={"embeddings": embeddings})

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self.retrieve(query).documents

    def retrieve(self, query: str) -> RetrievalResult:
        timings: dict[str, float] = {}

        started = time.perf_counter()
        query_vector = self.embeddings.embed_query(query)
        timings["embed_query"] = _elapsed_ms(started)

        started = time.perf_counter()
        dense = self.vector_store.max_marginal_relevance_search_by_vector(
            query_vector, k=self.candidate_k, fetch_k=self.candidate_k * 2
        )
        timings["dense_mmr"] = _elapsed_ms(started)

        started = time.perf_counter()
        tokens = tokenize(query)
        sparse = self._keyword_search(tokens)
        timings["bm25"] = _elapsed_ms(started)

        started = time.perf_counter()
        fused = reciprocal_rank_fusion_scored([dense, [doc for doc, _ in sparse]], rrf_k=self.rrf_k)
        timings["rrf_fusion"] = _elapsed_ms(started)

        return RetrievalResult(
            query=query,
            query_tokens=tokens,
            dense=dense,
            sparse=sparse,
            fused=fused,
            documents=[doc for doc, _ in fused[: self.k]],
            timings_ms=timings,
        )

    def _keyword_search(self, tokens: list[str]) -> list[tuple[Document, float]]:
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        return [(self.documents[i], float(scores[i])) for i in ranked[: self.candidate_k] if scores[i] > 0]


def reciprocal_rank_fusion(rankings: list[list[Document]], rrf_k: int) -> list[Document]:
    return [doc for doc, _ in reciprocal_rank_fusion_scored(rankings, rrf_k)]


def reciprocal_rank_fusion_scored(rankings: list[list[Document]], rrf_k: int) -> list[tuple[Document, float]]:
    scores: dict[tuple, float] = {}
    by_key: dict[tuple, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = document_key(doc)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            by_key.setdefault(key, doc)
    return [(by_key[key], scores[key]) for key in sorted(scores, key=scores.__getitem__, reverse=True)]


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def document_key(doc: Document) -> tuple:
    meta = doc.metadata
    return meta.get("path"), meta.get("start_line"), meta.get("end_line")


def tokenize(text: str) -> list[str]:
    tokens = []
    for word in _WORD.findall(text):
        tokens.append(word.lower())
        parts = _SUBWORD.findall(word)
        if len(parts) > 1:
            tokens.extend(part.lower() for part in parts)
    return tokens
