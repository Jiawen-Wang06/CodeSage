import fnmatch
import hashlib
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse


class RepoError(Exception):
    pass


EXCLUDED_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".venv", "venv",
    "node_modules", "bower_components", "vendor", "third_party",
    "dist", "build", "out", "target", "obj", ".next", ".nuxt", "coverage",
}

EXCLUDED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Pipfile.lock",
    "Cargo.lock", "composer.lock", "Gemfile.lock", "go.sum", "uv.lock",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".pdf",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar", ".jar", ".war",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a", ".class", ".pyc", ".whl",
    ".mp3", ".mp4", ".wav", ".mov", ".avi", ".woff", ".woff2", ".ttf", ".eot",
    ".db", ".sqlite", ".parquet", ".pkl", ".npy", ".bin", ".onnx", ".pt",
}

SECRET_PATTERNS = (
    ".env", ".env.*", "*.pem", "*.key", "*.p12", "*.pfx", "*.jks", "*.keystore",
    "id_rsa*", "id_dsa*", "id_ecdsa*", "id_ed25519*", ".npmrc", ".pypirc", ".netrc",
    "credentials*.json", "secrets.*", "*.tfstate", "*.tfvars",
)

GENERATED_PATTERNS = ("*.min.js", "*.min.css", "*.map", "*.bundle.js")

_REF_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/\-]*$")


@dataclass(frozen=True)
class RepoSnapshot:
    url: str
    path: Path
    commit: str

    @property
    def repo_id(self) -> str:
        return repo_id(self.url)

    @property
    def key(self) -> str:
        return f"{self.repo_id}_{self.commit[:12]}"


@dataclass(frozen=True)
class SourceFile:
    path: str
    text: str


def normalize_repo_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise RepoError("Only https:// repository URLs are supported.")
    if parsed.username or parsed.password:
        raise RepoError("Do not put credentials in the repository URL.")
    if parsed.query or parsed.fragment:
        raise RepoError("The repository URL must not contain a query string or fragment.")
    return url.rstrip("/")


def validate_ref(ref: str | None) -> str | None:
    if not ref:
        return None
    ref = ref.strip()
    if not _REF_PATTERN.match(ref) or ".." in ref:
        raise RepoError(f"Invalid branch or tag name: {ref!r}")
    return ref


def repo_id(url: str) -> str:
    return hashlib.sha256(url.lower().removesuffix(".git").encode()).hexdigest()[:16]


def clone_or_update(url: str, ref: str | None, repos_dir: Path, timeout_s: int) -> RepoSnapshot:
    url = normalize_repo_url(url)
    ref = validate_ref(ref)
    dest = repos_dir / repo_id(url)

    if (dest / ".git").is_dir():
        _run_git(["fetch", "--depth", "1", "origin", ref or "HEAD"], cwd=dest, timeout_s=timeout_s)
        _run_git(["checkout", "--force", "--detach", "FETCH_HEAD"], cwd=dest, timeout_s=timeout_s)
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        args = ["clone", "--depth", "1", "--single-branch", "-c", "core.symlinks=false"]
        if ref:
            args += ["--branch", ref]
        try:
            _run_git([*args, "--", url, str(dest)], timeout_s=timeout_s)
        except RepoError:
            remove_tree(dest)
            raise

    commit = _run_git(["rev-parse", "HEAD"], cwd=dest, timeout_s=timeout_s).strip()
    return RepoSnapshot(url=url, path=dest, commit=commit)


def list_source_files(snapshot: RepoSnapshot, max_file_bytes: int, max_files: int) -> list[SourceFile]:
    root = snapshot.path.resolve()
    listing = _run_git(["ls-files", "-z"], cwd=root, timeout_s=60)
    files: list[SourceFile] = []

    for rel_path in filter(None, listing.split("\0")):
        if is_excluded(rel_path):
            continue
        full_path = root / rel_path
        if full_path.is_symlink() or not full_path.is_file():
            continue
        if not full_path.resolve().is_relative_to(root):
            continue
        if full_path.stat().st_size > max_file_bytes:
            continue
        text = read_text(full_path)
        if text is None:
            continue
        files.append(SourceFile(path=rel_path, text=text))
        if len(files) > max_files:
            raise RepoError(f"Repository has more than {max_files} indexable files.")

    if not files:
        raise RepoError("No indexable source files were found in the repository.")
    return files


def is_excluded(rel_path: str) -> bool:
    path = PurePosixPath(rel_path)
    name = path.name
    if any(part in EXCLUDED_DIRS for part in path.parts[:-1]):
        return True
    if name in EXCLUDED_FILES or path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    return any(fnmatch.fnmatch(name.lower(), pattern) for pattern in SECRET_PATTERNS + GENERATED_PATTERNS)


def read_text(path: Path) -> str | None:
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    return text if text.strip() else None


def remove_tree(path: Path) -> None:
    def _make_writable_and_retry(func, target, _exc):
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if path.exists():
        shutil.rmtree(path, onexc=_make_writable_and_retry)


def _run_git(args: list[str], timeout_s: int, cwd: Path | None = None) -> str:
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GCM_INTERACTIVE": "never",
        "GIT_LFS_SKIP_SMUDGE": "1",
    }
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RepoError("git is not installed or not on PATH.") from exc
    except subprocess.TimeoutExpired as exc:
        raise RepoError(f"git {args[0]} timed out after {timeout_s}s.") from exc

    if result.returncode != 0:
        message = result.stderr.strip().splitlines()
        raise RepoError(f"git {args[0]} failed: {message[-1] if message else 'unknown error'}")
    return result.stdout
