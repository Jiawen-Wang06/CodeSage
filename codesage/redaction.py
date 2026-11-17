import re

REDACTED = "[REDACTED]"

_PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL,
)

_TOKEN_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}"),
    re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
]

_ASSIGNMENT = re.compile(
    r"""(?ix)
    (\b[\w.\-]*(?:api[_\-]?key|secret|passw(?:or)?d|pwd|token|access[_\-]?key|private[_\-]?key|client[_\-]?secret)[\w.\-]*
    ["']?\s*[:=]\s*["'])
    ([^"'\s]{8,})
    (["'])
    """
)

_CONNECTION_STRING_PASSWORD = re.compile(r"(\b[a-z][a-z0-9+.\-]*://[^:/\s@]+:)([^@\s/]+)(@)", re.IGNORECASE)


def redact_secrets(text: str) -> str:
    text = _PRIVATE_KEY_BLOCK.sub(_redact_keep_lines, text)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub(REDACTED, text)
    text = _ASSIGNMENT.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(3)}", text)
    return _CONNECTION_STRING_PASSWORD.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(3)}", text)


def _redact_keep_lines(match: re.Match) -> str:
    return REDACTED + "\n" * match.group(0).count("\n")
