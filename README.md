<p align="center">
  <img src="media/header.png" alt="CodeSage" width="720">
</p>

<p align="center">
  <b>Chat with any public GitHub repository.</b><br>
  A RAG (Retrieval-Augmented Generation) code assistant that indexes a codebase and answers questions about it with file and line citations.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white">
  <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white">
  <img alt="Chroma" src="https://img.shields.io/badge/Vector%20DB-Chroma-FF6F61">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="RAG" src="https://img.shields.io/badge/RAG-Hybrid%20Retrieval-6B4FBB">
</p>

---

## Demo

<p align="center">
  <img src="media/demo.gif" alt="CodeSage demo: indexing a repository and asking questions about it" width="820">
</p>

## What it does

Paste the URL of a public Git repository and CodeSage will:

1. **Clone** it (shallow) and pick out the files worth reading.
2. **Chunk** the code along language boundaries such as classes and functions.
3. **Index** it with both semantic embeddings and keyword search.
4. **Answer** your questions in a chat, citing the exact `path:start-end` lines it used, with the source snippets one click away.

Follow-up questions like *"where is **it** tested?"* work because the conversation is used to rewrite each question before searching.

## Why it is built this way

General-purpose RAG demos split text into fixed-size chunks and run a single vector search. That works poorly on code, so CodeSage makes a few deliberate choices:

| Problem with naive RAG on code | What CodeSage does |
|---|---|
| Fixed-size chunks cut functions in half | Language-aware splitting for 30+ extensions, ~1,500 chars with overlap |
| Embeddings are weak at exact identifiers like `parse_config` | **Hybrid retrieval**: dense MMR search + BM25 (with camelCase/snake_case sub-tokens), merged by Reciprocal Rank Fusion |
| Answers can't be verified | Every chunk carries `path`, `start_line`, `end_line`, and the model must cite them |
| "What does this project do?" retrieves random snippets | A synthetic **file-tree document** is indexed alongside the code |
| Follow-up questions lose their subject | An LLM step rewrites each follow-up into a standalone search query |
| Re-indexing on every question or commit is slow and costly | Persistent index per commit; on a new commit, **only changed files are re-embedded** |

### Incremental re-indexing

Each chunk stores a content hash of its file. When a repo moves to a new commit, CodeSage reuses the stored embeddings of every unchanged file and embeds only what changed, then prunes the old index. A shallow clone has no history to `git diff` against, so comparing content hashes is what makes this work.

Example from development: moving `pallets/itsdangerous` from `2.1.2` to `2.2.0` embedded **96 of 125** chunks and reused the rest. Re-opening the same commit loads instantly from cache.

## RAG architecture

Indexing happens once per commit. Answering runs a retrieve → augment → generate loop for each question.

```mermaid
flowchart LR
    U[User] -->|repo URL| UI[Streamlit UI]
    UI --> P[pipeline.py]

    subgraph Indexing
        P --> I[ingest.py<br/>validate · shallow clone · filter files]
        I --> R[redaction.py<br/>mask secrets]
        R --> C[chunking.py<br/>language-aware chunks + line metadata]
        C --> S[(index.py<br/>Chroma per commit<br/>+ embedding reuse)]
    end

    subgraph Answering
        UI -->|question + history| Q[rag.py]
        Q -->|standalone query| H[HybridRetriever<br/>MMR + BM25 → RRF]
        S --> H
        H -->|cited context| L[LLM]
        L -->|answer + sources| UI
    end
```

## Security and safety

The tool reads code written by strangers and sends it to an LLM, so it is built with that in mind:

- **Bring your own key.** Each user enters their own OpenAI key in the sidebar. It lives only in that user's session memory, is passed explicitly to the OpenAI clients, and is never written to disk, logged, or shown back. OpenAI error messages (which can echo part of a key) are replaced with generic ones. In deployment, `CODESAGE_BYOK_ONLY=true` guarantees the server never falls back to a key of its own.
- **Secrets in the repo stay out of the index.** Files such as `.env`, `*.pem` and `id_rsa` are skipped. Common token formats (OpenAI, GitHub, AWS, Google, Slack, Stripe, JWTs, private-key blocks, password assignments, connection-string passwords) are replaced with `[REDACTED]` **before** embedding, so they never reach the vector store or the model.
- **Safe cloning.** Only `https://` URLs are accepted, and URLs with embedded credentials are rejected. Branch names are validated against option injection. Clones are shallow and time-limited, with symlinks, Git LFS downloads and interactive prompts turned off. Nothing from the repository is ever executed.
- **Prompt-injection aware.** Repository content is delimited and treated as untrusted data in the system prompt, and the assistant refuses requests for credentials.
- **Public repositories only**, by design.

## Quick start

**Requirements:** Python 3.12+, Git on your `PATH`, and an OpenAI API key.

```bash
git clone https://github.com/Jiawen-Wang06/CodeSage.git
cd CodeSage

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

Run the app:

```bash
streamlit run code_sage.py
```

Enter your OpenAI API key in the sidebar, paste a repository URL (for example `https://github.com/Jiawen-Wang06/ElderSafe`), click **Index repository**, and start asking questions.

For local development you can skip the sidebar by putting the key in a git-ignored `.env` file:

```env
OPENAI_API_KEY=your-key-here
```

### Optional configuration

| Variable | Default | Purpose |
|---|---|---|
| `CODESAGE_CHAT_MODEL` | `gpt-4o-mini` | Model used for answers and query rewriting |
| `CODESAGE_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model (changing it triggers a re-index) |
| `CODESAGE_DATA_DIR` | `.codesage` | Where clones and indexes are cached |
| `CODESAGE_BYOK_ONLY` | `false` | When `true`, users must enter their own key and the server's `OPENAI_API_KEY` is ignored |
| `CODESAGE_TRACE` | `false` | When `true`, every question is written as an end-to-end trace (see below) |
| `CODESAGE_TRACE_DIR` | `<data dir>/traces` | Where trace files are written |
| `CODESAGE_TRACE_RETENTION_DAYS` | unset (keep all) | Delete trace files older than this many days |
| `CODESAGE_TRACE_SHOW_ID` | `false` | Show the trace id under each answer. For local debugging only; leave unset in deployment |

Chunk size, retrieval depth, file limits and chat-history length live in `codesage/config.py`.

### Tracing for evaluation

With `CODESAGE_TRACE=true`, each question is appended as one JSON line to `traces/traces-YYYY-MM-DD-<instance>.jsonl` (one file per server instance, so scaled-out instances never write to the same file). The sidebar tells users that conversations are logged. Nothing about tracing is shown to users unless `CODESAGE_TRACE_SHOW_ID=true` is also set. A trace holds the input, every step, and the output:

```
input            question + full conversation history
steps
  history_window     which turns were kept for the prompt
  condense_question  LLM: rendered prompt, rewritten query, model, tokens  (skipped on the first turn)
  retrieve           final chunks with text, RRF score, dense and BM25 ranks
    embed_query        embedding latency
    dense_mmr_search   vector candidates (rank, path, lines)
    bm25_search        query tokens, keyword candidates with BM25 scores
    rrf_fusion         fused ranking, which chunks were selected
  generate_answer    LLM: exact messages sent, answer text, model, finish reason, tokens
  check_citations    cited vs retrieved: hallucinated and unused sources
output           answer, search query, sources, citations
```

Every step also has `status`, `started_at` and `duration_ms`. Failed requests are written too, with the failing step marked `error`. Use `codesage.tracing.read_traces(dir)` to load them.

Privacy: API keys are never passed into a trace, and the active key is scrubbed if it appears anywhere (e.g. pasted into a question). All strings are run through the same secret redaction used for indexing, plus e-mail masking, before they are written. Errors record only the exception type, never the message.

## Project structure

```
code_sage.py            Streamlit chat UI with the bring-your-own-key sidebar
Dockerfile              Production image for Azure App Service
codesage/
  config.py             Settings and environment overrides
  ingest.py             URL validation, shallow clone, file filtering
  redaction.py          Secret masking before anything is embedded
  chunking.py           Language-aware chunking with line-range metadata
  index.py              Persistent Chroma store, embedding reuse, pruning, hybrid retriever
  rag.py                Prompts, follow-up query rewriting, cited answers
  pipeline.py           Orchestrates indexing and builds the QA chain
  tracing.py            Opt-in end-to-end request traces (JSONL) for evaluation
```

Each stage sits behind a small interface, so swapping the vector store, LLM provider or chunking strategy touches a single module.

## Tech stack

- **LLM and embeddings:** OpenAI (`gpt-4o-mini`, `text-embedding-3-small`) via LangChain
- **Vector store:** Chroma (persistent)
- **Keyword search:** `rank_bm25`
- **UI:** Streamlit 
- **Ingestion:** Git CLI
- **Deployment:** Docker, Azure App Service, Azure Container Registry

