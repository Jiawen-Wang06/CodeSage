import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_TRUE_VALUES = {"1", "true", "yes", "on"}


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class Settings:
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    data_dir: Path = Path(".codesage")
    byok_only: bool = False
    chunk_size: int = 1500
    chunk_overlap: int = 200
    max_file_bytes: int = 500_000
    max_files: int = 5_000
    git_timeout_s: int = 300
    retrieval_k: int = 8
    embed_batch_size: int = 128
    history_messages: int = 8
    trace_enabled: bool = False
    trace_dir: Path | None = None
    trace_show_id: bool = False
    trace_retention_days: int | None = None

    @property
    def repos_dir(self) -> Path:
        return self.data_dir / "repos"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def traces_dir(self) -> Path:
        return self.trace_dir or self.data_dir / "traces"


def load_settings() -> Settings:
    load_dotenv()
    defaults = Settings()
    return Settings(
        chat_model=os.getenv("CODESAGE_CHAT_MODEL", defaults.chat_model),
        embedding_model=os.getenv("CODESAGE_EMBEDDING_MODEL", defaults.embedding_model),
        data_dir=Path(os.getenv("CODESAGE_DATA_DIR", str(defaults.data_dir))),
        byok_only=os.getenv("CODESAGE_BYOK_ONLY", "").strip().lower() in _TRUE_VALUES,
        trace_enabled=os.getenv("CODESAGE_TRACE", "").strip().lower() in _TRUE_VALUES,
        trace_show_id=os.getenv("CODESAGE_TRACE_SHOW_ID", "").strip().lower() in _TRUE_VALUES,
        trace_retention_days=_positive_int(os.getenv("CODESAGE_TRACE_RETENTION_DAYS")),
        trace_dir=Path(trace_dir) if (trace_dir := os.getenv("CODESAGE_TRACE_DIR", "").strip()) else None,
    )


def _positive_int(value: str | None) -> int | None:
    try:
        number = int((value or "").strip())
    except ValueError:
        return None
    return number if number > 0 else None


def server_key_available(settings: Settings) -> bool:
    return not settings.byok_only and bool(os.getenv("OPENAI_API_KEY"))


def resolve_api_key(user_key: str | None, settings: Settings) -> str:
    key = (user_key or "").strip()
    if key:
        if any(char.isspace() for char in key):
            raise ConfigError("The OpenAI API key contains whitespace. Paste it again.")
        return key
    if server_key_available(settings):
        return os.environ["OPENAI_API_KEY"]
    raise ConfigError("Enter your OpenAI API key in the sidebar to continue.")
