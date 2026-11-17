import hashlib
from pathlib import PurePosixPath

from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from codesage.ingest import RepoSnapshot, SourceFile
from codesage.redaction import redact_secrets

CHUNKER_VERSION = 1

EXTENSION_LANGUAGES: dict[str, Language] = {
    ".py": Language.PYTHON,
    ".js": Language.JS, ".jsx": Language.JS, ".mjs": Language.JS, ".cjs": Language.JS,
    ".ts": Language.TS, ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN, ".kts": Language.KOTLIN,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".rb": Language.RUBY,
    ".php": Language.PHP,
    ".cs": Language.CSHARP,
    ".c": Language.C, ".h": Language.C,
    ".cpp": Language.CPP, ".cc": Language.CPP, ".cxx": Language.CPP, ".hpp": Language.CPP,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".lua": Language.LUA,
    ".pl": Language.PERL,
    ".hs": Language.HASKELL,
    ".ex": Language.ELIXIR, ".exs": Language.ELIXIR,
    ".ps1": Language.POWERSHELL,
    ".r": Language.R,
    ".sol": Language.SOL,
    ".proto": Language.PROTO,
    ".md": Language.MARKDOWN, ".markdown": Language.MARKDOWN,
    ".rst": Language.RST,
    ".tex": Language.LATEX,
    ".html": Language.HTML, ".htm": Language.HTML,
}

FILE_TREE_PATH = "<repository file tree>"


class CodeChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitters: dict[Language | None, RecursiveCharacterTextSplitter] = {}

    def chunk_repository(self, snapshot: RepoSnapshot, files: list[SourceFile]) -> list[Document]:
        documents = self.file_tree_documents(snapshot, files)
        for source in files:
            documents.extend(self.chunk_file(snapshot, source))
        return documents

    def chunk_file(self, snapshot: RepoSnapshot, source: SourceFile) -> list[Document]:
        language = detect_language(source.path)
        language_name = language.value if language else "text"
        file_hash = content_hash(source.text)
        return [
            Document(
                page_content=f"File: {source.path}\n{piece}",
                metadata=_metadata(snapshot, source.path, language_name, piece, start_line, file_hash),
            )
            for piece, start_line in _split_with_lines(self._splitter_for(language), redact_secrets(source.text))
        ]

    def file_tree_documents(self, snapshot: RepoSnapshot, files: list[SourceFile]) -> list[Document]:
        tree = "\n".join(sorted(source.path for source in files))
        return [
            Document(
                page_content=f"Files in repository {snapshot.url}:\n{piece}",
                metadata=_metadata(snapshot, FILE_TREE_PATH, "text", piece, start_line, content_hash(tree)),
            )
            for piece, start_line in _split_with_lines(self._splitter_for(None), tree)
        ]

    def _splitter_for(self, language: Language | None) -> RecursiveCharacterTextSplitter:
        if language not in self._splitters:
            if language is None:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
                )
            else:
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap
                )
            self._splitters[language] = splitter
        return self._splitters[language]


def detect_language(path: str) -> Language | None:
    return EXTENSION_LANGUAGES.get(PurePosixPath(path).suffix.lower())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_with_lines(splitter: RecursiveCharacterTextSplitter, text: str) -> list[tuple[str, int]]:
    pieces = []
    cursor = 0
    for piece in splitter.split_text(text):
        start = text.find(piece, cursor)
        if start == -1:
            start = max(text.find(piece), 0)
        pieces.append((piece, text.count("\n", 0, start) + 1))
        cursor = start + 1
    return pieces


def _metadata(
    snapshot: RepoSnapshot, path: str, language: str, piece: str, start_line: int, file_hash: str
) -> dict:
    return {
        "path": path,
        "language": language,
        "start_line": start_line,
        "end_line": start_line + piece.count("\n"),
        "file_hash": file_hash,
        "commit": snapshot.commit,
        "repo": snapshot.url,
    }
