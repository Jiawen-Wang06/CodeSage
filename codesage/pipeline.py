import threading
from dataclasses import dataclass, replace
from typing import Callable, Sequence

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from codesage.chunking import CHUNKER_VERSION, CodeChunker, content_hash
from codesage.config import Settings
from codesage.index import HybridRetriever, IndexStore, RepoIndex, StoredChunk
from codesage.ingest import RepoSnapshot, SourceFile, clone_or_update, list_source_files, normalize_repo_url, repo_id
from codesage.rag import Answer, ChatTurn, CodeQA
from codesage.tracing import Trace, write_trace

ProgressReporter = Callable[[str, float], None]

_repo_locks: dict[str, threading.Lock] = {}
_repo_locks_guard = threading.Lock()


@dataclass
class IndexedRepo:
    snapshot: RepoSnapshot
    index: RepoIndex
    retriever: HybridRetriever
    chunk_count: int
    embedded_count: int
    from_cache: bool

    def close(self) -> None:
        self.index.close()


def index_repository(
    url: str,
    ref: str | None,
    settings: Settings,
    api_key: str,
    report: ProgressReporter,
    replacing: IndexedRepo | None = None,
) -> IndexedRepo:
    url = normalize_repo_url(url)
    embeddings = build_embeddings(settings, api_key)

    report("Waiting for repository lock", 0.0)
    with _repo_lock(repo_id(url)):
        report("Fetching repository", 0.02)
        snapshot = clone_or_update(url, ref, settings.repos_dir, settings.git_timeout_s)
        store = build_index_store(settings, embeddings)
        from_cache = store.exists(snapshot.key)

        if from_cache:
            report("Loading cached index", 0.5)
            index = store.load(snapshot.key)
            embedded_count = 0
        else:
            index, embedded_count = _build_index(snapshot, store, settings, report)

        if replacing is not None:
            replacing.close()
        store.prune(snapshot.repo_id, keep=snapshot.key)

    report("Building keyword index", 0.97)
    retriever = HybridRetriever.from_index(index, embeddings, k=settings.retrieval_k)
    report("Ready", 1.0)
    return IndexedRepo(
        snapshot=snapshot,
        index=index,
        retriever=retriever,
        chunk_count=len(index.documents),
        embedded_count=embedded_count,
        from_cache=from_cache,
    )


def build_qa(repo: IndexedRepo, settings: Settings, api_key: str) -> CodeQA:
    retriever = repo.retriever.with_embeddings(build_embeddings(settings, api_key))
    return CodeQA(retriever, build_llm(settings, api_key), settings.history_messages)


def answer_question(
    repo: IndexedRepo,
    settings: Settings,
    api_key: str,
    question: str,
    history: Sequence[ChatTurn],
    session_id: str,
) -> Answer:
    qa = build_qa(repo, settings, api_key)
    trace = Trace(
        session_id=session_id,
        turn_index=sum(1 for turn in history if turn.role == "user"),
        repo={"url": repo.snapshot.url, "commit": repo.snapshot.commit, "chunk_count": repo.chunk_count},
        config=trace_config(settings, repo.retriever),
        scrub=(api_key,),
    )
    try:
        return qa.ask(question, history=history, trace=trace)
    except Exception as exc:
        trace.fail(exc)
        raise
    finally:
        if settings.trace_enabled:
            write_trace(trace, settings.traces_dir, settings.trace_retention_days)


def trace_config(settings: Settings, retriever: HybridRetriever) -> dict:
    return {
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "temperature": 0,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunker_version": CHUNKER_VERSION,
        "history_messages": settings.history_messages,
        "retrieval_k": retriever.k,
        "candidate_k": retriever.candidate_k,
        "rrf_k": retriever.rrf_k,
    }


def build_index_store(settings: Settings, embeddings: OpenAIEmbeddings) -> IndexStore:
    fingerprint = {
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "chunker_version": CHUNKER_VERSION,
    }
    return IndexStore(settings.index_dir, embeddings, settings.embed_batch_size, fingerprint)


def build_embeddings(settings: Settings, api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=api_key)


def build_llm(settings: Settings, api_key: str) -> ChatOpenAI:
    return ChatOpenAI(model=settings.chat_model, temperature=0, api_key=api_key)


def _repo_lock(key: str) -> threading.Lock:
    with _repo_locks_guard:
        return _repo_locks.setdefault(key, threading.Lock())


def _build_index(
    snapshot: RepoSnapshot, store: IndexStore, settings: Settings, report: ProgressReporter
) -> tuple[RepoIndex, int]:
    report("Scanning files", 0.05)
    files = list_source_files(snapshot, settings.max_file_bytes, settings.max_files)

    previous = store.latest(snapshot.repo_id, exclude=snapshot.key)
    if previous:
        report("Comparing with previous index", 0.08)
    stored = store.stored_chunks(previous) if previous else {}

    report(f"Chunking {len(files)} files", 0.1)
    reused, new_documents = _plan_chunks(snapshot, files, stored, CodeChunker(settings.chunk_size, settings.chunk_overlap))

    index = store.build(
        snapshot.key,
        new_documents,
        reused,
        lambda done, total: report(f"Embedding chunks {done}/{total}", 0.1 + 0.85 * done / total),
    )
    return index, len(new_documents)


def _plan_chunks(
    snapshot: RepoSnapshot,
    files: list[SourceFile],
    stored: dict[tuple[str, str], list[StoredChunk]],
    chunker: CodeChunker,
) -> tuple[list[StoredChunk], list[Document]]:
    reused: list[StoredChunk] = []
    new_documents = chunker.file_tree_documents(snapshot, files)
    for source in files:
        previous_chunks = stored.get((source.path, content_hash(source.text)))
        if previous_chunks:
            reused.extend(_retarget(chunk, snapshot) for chunk in previous_chunks)
        else:
            new_documents.extend(chunker.chunk_file(snapshot, source))
    return reused, new_documents


def _retarget(chunk: StoredChunk, snapshot: RepoSnapshot) -> StoredChunk:
    document = Document(page_content=chunk.document.page_content, metadata={**chunk.document.metadata, "commit": snapshot.commit})
    return replace(chunk, document=document)
