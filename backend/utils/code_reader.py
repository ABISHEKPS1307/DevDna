"""DevDNA – repository scanning: safe ZIP extraction, language/framework
detection, endpoint counting and code-file discovery."""

from __future__ import annotations

import logging
import posixpath
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("devdna.code_reader")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

IGNORED_DIRS = {
    "venv", ".venv", "env", "node_modules", ".git", "__pycache__",
    ".vscode", ".idea", "dist", "build", ".devdna", ".next", "coverage",
    "migrations", ".pytest_cache", ".mypy_cache", ".github",
}

LANGUAGE_MAP = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript",
    ".java": "Java", ".html": "HTML", ".htm": "HTML", ".css": "CSS",
    ".ts": "TypeScript", ".tsx": "React TSX", ".jsx": "React JSX",
}

# Hard safety limits (defence against hostile / bloated archives)
MAX_UPLOAD_BYTES = 100 * 1024 * 1024       # 100 MB compressed
MAX_EXTRACTED_BYTES = 400 * 1024 * 1024    # 400 MB uncompressed
MAX_SINGLE_FILE_BYTES = 2 * 1024 * 1024    # 2 MB per source file
MAX_FILES = 8_000
MAX_DEPTH = 12

_FRAMEWORK_SIGNATURES = [
    ("FastAPI", (r"\bFastAPI\s*\(", r"\bfastapi\b")),
    ("Flask", (r"\bFlask\s*\(", r"\bfrom\s+flask\b")),
    ("Django", (r"\bfrom\s+django\b", r"\bdjango\.conf\b")),
    ("Express", (r"require\(\s*['\"]express['\"]\s*\)", r"from\s+['\"]express['\"]")),
    ("Spring Boot", (r"@SpringBootApplication", r"org\.springframework")),
    ("React", (r"from\s+['\"]react['\"]", r"require\(\s*['\"]react['\"]\s*\)")),
]

# Heuristic endpoint counting: Python decorators and Express-style routes.
_PY_ENDPOINT_RE = re.compile(r"@\s*\w[\w.]*\.(get|post|put|delete|patch)\s*\(")
_JS_ENDPOINT_RE = re.compile(
    r"\b(?:app|router|server|api)\s*\.\s*(get|post|put|delete|patch)\s*\("
)

_BRANCH_SUFFIX_RE = re.compile(r"[-_](main|master|develop|dev|release)$", re.IGNORECASE)


class InvalidZipError(ValueError):
    """Raised when the archive is corrupt, hostile, or has no usable content."""


# --------------------------------------------------------------------------- #
# Safe extraction
# --------------------------------------------------------------------------- #

def _safe_target(dest_root: Path, member_name: str) -> Optional[Path]:
    """Map a zip member to a path strictly inside dest_root (zip-slip guard)."""
    name = member_name.replace("\\", "/")
    if name.startswith("/") or ".." in name.split("/"):
        return None
    name = posixpath.normpath(name)
    if name in ("", "."):
        return None
    root = dest_root.resolve()
    target = (root / name).resolve()
    if not target.is_relative_to(root):  # Python 3.9+
        return None
    return target


def safe_extract(zip_path: Path, dest_root: Path) -> None:
    """Extract a ZIP manually with zip-slip / zip-bomb / symlink guards."""
    dest_root.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    file_count = 0
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                # Skip symlinks and non-regular entries
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    continue
                if info.is_dir():
                    target = _safe_target(dest_root, info.filename)
                    if target:
                        target.mkdir(parents=True, exist_ok=True)
                    continue
                depth = len(info.filename.replace("\\", "/").strip("/").split("/"))
                if depth > MAX_DEPTH:
                    continue
                target = _safe_target(dest_root, info.filename)
                if target is None:
                    logger.warning("Skipping unsafe zip member: %s", info.filename)
                    continue
                if file_count >= MAX_FILES:
                    raise InvalidZipError("Archive contains too many files.")
                total_bytes += info.file_size
                if total_bytes > MAX_EXTRACTED_BYTES:
                    raise InvalidZipError("Archive is too large when extracted.")
                target.parent.mkdir(parents=True, exist_ok=True)
                copied = 0
                with zf.open(info) as src, open(target, "wb") as out:
                    while True:
                        block = src.read(1024 * 1024)
                        if not block:
                            break
                        copied += len(block)
                        if copied > MAX_SINGLE_FILE_BYTES:
                            break  # truncate oversized files
                        out.write(block)
                file_count += 1
    except zipfile.BadZipFile:
        raise
    except OSError as exc:
        raise InvalidZipError(f"Could not extract archive safely: {exc}") from exc


def find_repo_root(dest_root: Path) -> Path:
    """If extraction produced one top-level folder (GitHub ZIP), use it."""
    entries = list(dest_root.iterdir())
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return dest_root


def derive_repo_name(zip_stem: str, repo_root: Path, dest_root: Path) -> str:
    """'APIFlow-main' -> 'APIFlow'."""
    raw = zip_stem if repo_root == dest_root else repo_root.name
    return _BRANCH_SUFFIX_RE.sub("", raw).strip("-_ .") or "repository"


# --------------------------------------------------------------------------- #
# Scanning
# --------------------------------------------------------------------------- #

def detect_language(path: Path) -> Optional[str]:
    return LANGUAGE_MAP.get(path.suffix.lower())


def iter_code_files(root: Path) -> Iterator[Tuple[Path, str, str]]:
    """Yield (abs_path, relative_posix_path, language) for supported files."""
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts[:-1]):
            continue
        if path.name.startswith("."):
            continue
        lang = detect_language(path)
        if not lang:
            continue
        try:
            if path.stat().st_size > MAX_SINGLE_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path, rel.as_posix(), lang


def detect_framework(code_texts: List[Tuple[str, str]]) -> str:
    for framework, patterns in _FRAMEWORK_SIGNATURES:
        for _path, content in code_texts:
            for pattern in patterns:
                if re.search(pattern, content):
                    return framework
    return "Unknown"


def count_endpoints(code_texts: List[Tuple[str, str]]) -> int:
    count = 0
    for path, content in code_texts:
        if path.endswith(".py"):
            count += len(_PY_ENDPOINT_RE.findall(content))
        elif path.endswith((".js", ".ts", ".jsx", ".tsx", ".mjs")):
            count += len(_JS_ENDPOINT_RE.findall(content))
    return count


def scan_repository(repo_root: Path) -> dict:
    """Walk the repository and compute the DNA report data."""
    code_texts: List[Tuple[str, str]] = []
    languages: Dict[str, int] = {}
    detected_files: List[str] = []

    for path, rel, lang in iter_code_files(repo_root):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        code_texts.append((rel, content))
        languages[lang] = languages.get(lang, 0) + 1
        detected_files.append(rel)

    return {
        "code_texts": code_texts,
        "languages": languages,
        "detected_files": detected_files,
        "total_code_files": len(code_texts),
        "framework": detect_framework(code_texts),
        "endpoints": count_endpoints(code_texts),
    }
