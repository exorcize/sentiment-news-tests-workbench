import hashlib
import re
import unicodedata


_PREFIX_RE = re.compile(
    r"^\s*(?:update|breaking|alert|exclusive|watch|flash|reuters|bloomberg|wsj)\s*[:\-]\s*",
    re.IGNORECASE,
)
_QUOTES_RE = re.compile(r"[‘’“”`']")
_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Strip common news prefixes, smart quotes, collapse whitespace, lowercase."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    # Strip up to 3 chained prefixes (e.g. "UPDATE: BREAKING: ...")
    for _ in range(3):
        new = _PREFIX_RE.sub("", out)
        if new == out:
            break
        out = new
    out = _QUOTES_RE.sub('"', out)
    out = _WHITESPACE_RE.sub(" ", out).strip().lower()
    return out


def cache_key(text: str, symbol: str | None) -> str:
    """Build a deterministic cache key for sentiment results."""
    norm = normalize_text(text)
    sym = (symbol or "").upper().strip()
    digest = hashlib.sha1(f"{sym}|{norm}".encode("utf-8")).hexdigest()
    return f"sentiment:v1:{digest}"


def text_hash(text: str) -> str:
    """Stable hash of normalized text (for dataset dedup)."""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()
