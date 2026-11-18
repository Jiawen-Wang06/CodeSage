import uuid
from pathlib import Path

import openai
import streamlit as st

from codesage.config import ConfigError, Settings, load_settings, resolve_api_key, server_key_available
from codesage.ingest import RepoError
from codesage.pipeline import IndexedRepo, answer_question, index_repository
from codesage.rag import ChatTurn, chunk_body, source_label

LOGO_PATH = Path(__file__).parent / "codeSage.png"
OPENAI_KEYS_URL = "https://platform.openai.com/api-keys"


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, (RepoError, ConfigError)):
        return str(exc)
    if isinstance(exc, openai.AuthenticationError):
        return "OpenAI rejected the API key. Check the key in the sidebar."
    if isinstance(exc, openai.PermissionDeniedError):
        return "This API key does not have access to the required OpenAI models."
    if isinstance(exc, openai.RateLimitError):
        return "OpenAI rate limit or quota reached for this key. Try again later or check your billing."
    if isinstance(exc, openai.APIError):
        return f"OpenAI request failed ({type(exc).__name__})."
    return f"Unexpected error ({type(exc).__name__})."


def render_header() -> None:
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width="stretch")


def render_sidebar(settings: Settings) -> str | None:
    with st.sidebar:
        st.subheader("Your OpenAI API key")
        user_key = st.text_input(
            "OpenAI API key",
            type="password",
            key="openai_api_key",
            placeholder="sk-...",
            label_visibility="collapsed",
        )
        st.caption(
            "Used only for this browser session to call OpenAI. It is held in server memory, "
            "never written to disk or logs, and discarded when you close the tab."
        )
        st.caption(f"Tip: create a dedicated key with a spending limit at [platform.openai.com]({OPENAI_KEYS_URL}).")
        if not user_key and server_key_available(settings):
            st.caption("No key entered, using the server's OPENAI_API_KEY (local development).")
        if settings.trace_enabled:
            st.caption(
                "Questions, answers and the retrieved code are logged to evaluate and improve answer quality. "
                "Your API key is never logged. Don't paste secrets or personal information into the chat."
            )

        try:
            return resolve_api_key(user_key, settings)
        except ConfigError as exc:
            if user_key:
                st.error(str(exc))
            return None


def render_repo_form(settings: Settings, api_key: str | None) -> None:
    with st.form("repo_form", border=False):
        url = st.text_input("Public Git repository URL", placeholder="https://github.com/owner/repo")
        ref = st.text_input("Branch or tag (optional)", placeholder="main")
        submitted = st.form_submit_button("Index repository", disabled=api_key is None)

    if not submitted or api_key is None:
        return
    if not url.strip():
        st.warning("Enter a repository URL.")
        return

    progress = st.progress(0.0, text="Starting")
    try:
        repo = index_repository(
            url,
            ref,
            settings,
            api_key,
            lambda message, value: progress.progress(min(value, 1.0), text=message),
            replacing=st.session_state.get("repo"),
        )
        st.session_state.repo = repo
        st.session_state.messages = []
    except Exception as exc:
        progress.empty()
        st.error(friendly_error(exc))


def render_repo_status(repo: IndexedRepo) -> None:
    if repo.from_cache:
        detail = "loaded from cache"
    else:
        reused = repo.chunk_count - repo.embedded_count
        detail = f"embedded {repo.embedded_count}, reused {reused}" if reused else "newly indexed"
    st.success(f"{repo.snapshot.url} @ {repo.snapshot.commit[:7]} · {repo.chunk_count} chunks ({detail})")


def render_sources(turn: ChatTurn) -> None:
    if not turn.sources:
        return
    with st.expander(f"Sources ({len(turn.sources)})"):
        if turn.search_query:
            st.caption(f"Searched for: {turn.search_query}")
        for doc in turn.sources:
            st.caption(source_label(doc))
            st.code(chunk_body(doc), language=doc.metadata.get("language"))


def render_turn(turn: ChatTurn) -> None:
    with st.chat_message(turn.role):
        st.markdown(turn.content)
        render_sources(turn)


def render_chat(repo: IndexedRepo, settings: Settings, api_key: str | None) -> None:
    messages: list[ChatTurn] = st.session_state.setdefault("messages", [])

    if messages and st.button("Clear chat"):
        messages.clear()

    for turn in messages:
        render_turn(turn)

    placeholder = "Ask a question about the code" if api_key else "Enter your OpenAI API key to ask questions"
    question = st.chat_input(placeholder, disabled=api_key is None)
    if not question or not question.strip() or api_key is None:
        return

    question = question.strip()
    user_turn = ChatTurn(role="user", content=question)
    render_turn(user_turn)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking"):
                session_id = st.session_state.setdefault("session_id", uuid.uuid4().hex)
                answer = answer_question(repo, settings, api_key, question, messages, session_id)
        except Exception as exc:
            st.error(friendly_error(exc))
            return
        assistant_turn = ChatTurn(
            role="assistant",
            content=answer.text,
            sources=answer.sources,
            search_query=answer.search_query if answer.search_query != question else None,
        )
        st.markdown(assistant_turn.content)
        render_sources(assistant_turn)
        if settings.trace_enabled and settings.trace_show_id and answer.trace_id:
            st.caption(f"Trace `{answer.trace_id}`")

    messages.extend([user_turn, assistant_turn])


def main() -> None:
    st.set_page_config(page_title="CodeSage", page_icon="🧭")
    settings = load_settings()
    api_key = render_sidebar(settings)
    render_header()

    if api_key is None:
        st.info("Enter your OpenAI API key in the sidebar to get started.")
    render_repo_form(settings, api_key)

    repo = st.session_state.get("repo")
    if repo is None:
        if api_key is not None:
            st.info("Index a public repository to start asking questions.")
        return

    render_repo_status(repo)
    render_chat(repo, settings, api_key)


main()
