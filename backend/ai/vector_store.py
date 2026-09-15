"""DevDNA – code chunking + persistent TF-IDF vector store with hybrid search,
plus file-tree support for the repository explorer."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("devdna.vector_store")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
INDEX_VERSION = 2
TOP_K = 3
MIN_SCORE = 0.05
MAX_PREVIEW_CHUNKS = 20

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "and", "or", "but", "if",
    "it", "its", "this", "that", "these", "those", "as", "by", "from",
    "how", "what", "where", "when", "which", "who", "why", "does", "do",
    "did", "can", "could", "should", "would", "will", "my", "your", "our",
    "me", "you", "we", "they", "he", "she", "his", "her", "their", "there",
    "have", "has", "had", "not", "no", "yes", "so", "than", "then", "too",
    "very", "just", "about", "into", "over", "under", "again", "all", "any",
    "code", "file", "files", "find", "show", "tell", "please", "explain",
}

SYNONYMS = {
    "uploaded": "upload", "uploading": "upload", "uploads": "upload",
    "downloading": "download", "downloaded": "download", "downloads": "download",
    "api": "fastapi", "apis": "fastapi",
    "route": "endpoint", "routes": "endpoint", "routing": "endpoint",
    "endpoints": "endpoint",
    "templates": "template",
    "authentication": "auth", "authenticate": "auth", "authenticating": "auth",
    "login": "auth", "signin": "auth",
    "db": "database",
}

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL_RE = re.compile(
    r".+?(?:(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z0-9])|$)"
)


def _stem(token: str) -> str:
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            if suffix == "ies":
                return token[:-3] + "y"
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    """Split identifiers into searchable tokens.

    Handles snake_case, kebab-case and camelCase, e.g.
    ``Jinja2Templates`` -> ['jinja2', 'template'],
    ``upload_excel``    -> ['upload', 'excel'].
    """
    tokens: List[str] = []
    for raw in _WORD_RE.findall(text):
        for part in re.split(r"[_\-\s]+", raw):
            if not part:
                continue
            if part.isupper() or part.islower() or part.isdigit():
                pieces = [part]
            else:
                pieces = [m.group(0) for m in _CAMEL_RE.finditer(part)]
            for piece in pieces:
                low = piece.lower()
                if len(low) < 2 or low in STOPWORDS:
                    continue
                tokens.append(_stem(SYNONYMS.get(low, low)))
    return tokens


def strip_repo_prefix(path: str, repository: str) -> str:
    """'APIFlow/main.py' -> 'main.py' (canonical repository-relative path)."""
    prefix = f"{repository}/" if repository else ""
    return path[len(prefix):] if prefix and path.startswith(prefix) else path


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def _fallback_chunks(text: str, size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        if end < n:
            nl = chunk.rfind("\n")
            if nl > size // 2:
                end = start + nl + 1
                chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)  # guarantee forward progress
    return chunks


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split source code into overlapping ~size-char chunks."""
    text = text.strip("\n")
    if not text.strip():
        return []
    size = max(50, size)
    overlap = max(0, min(overlap, size - 1))
    chunks: List[str] = []
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap, length_function=len,
        )
        chunks = [c for c in splitter.split_text(text) if c.strip()]
    except Exception:  # LangChain not installed – graceful fallback
        chunks = []
    if not chunks:
        chunks = _fallback_chunks(text, size, overlap)
    return chunks


def _build_tree(files: List[str]) -> List[dict]:
    """Convert a flat list of repo-relative paths into a nested node tree."""
    root: Dict[str, dict] = {}
    for rel in files:
        node = root
        parts = rel.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = None  # file marker

    def convert(branch: Dict[str, dict], prefix: str) -> List[dict]:
        children: List[dict] = []
        # folders first, then files, alphabetical
        for name, sub in sorted(branch.items(), key=lambda kv: (kv[1] is None, kv[0].lower())):
            path = f"{prefix}{name}"
            if sub is None:
                children.append({"type": "file", "name": name, "path": path})
            else:
                children.append({"type": "dir", "name": name, "path": path,
                                 "children": convert(sub, f"{path}/")})
        return children

    return convert(root, "")


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #

class DevDNAVectorStore:
    """In-memory TF-IDF store persisted as JSON, with hybrid search."""

    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._lock = threading.RLock()
        self._documents: List[dict] = []
        self._files: List[str] = []
        self._idf: Dict[str, float] = {}
        self._repository: str = ""

    @property
    def repository(self) -> str:
        return self._repository

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def ensure_loaded(self) -> bool:
        """Reload the persisted index if memory is empty."""
        with self._lock:
            if self._documents:
                return True
            if not self.index_path.exists():
                return False
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self._documents = data.get("documents", [])
                self._idf = data.get("idf", {})
                self._repository = data.get("repository", "")
                files = data.get("files")
                if not files:  # backfill for older indexes
                    files = sorted({strip_repo_prefix(d.get("path", ""), self._repository)
                                    for d in self._documents})
                self._files = files
                logger.info("Loaded index: %d chunks, %d files from %s",
                            len(self._documents), len(self._files), self.index_path)
                return bool(self._documents)
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("Failed to load index %s: %s", self.index_path, exc)
                return False

    def build(self, repository: str, files: List[dict]) -> int:
        """Chunk + vectorise files, persist atomically.
        files entries: {file, path (repo-prefixed), rel (repo-relative), content}"""
        with self._lock:
            docs: List[dict] = []
            rels = set()
            for entry in files:
                rel = entry.get("rel") or entry["path"]
                rels.add(rel)
                for i, piece in enumerate(chunk_text(entry["content"]), start=1):
                    docs.append({
                        "file": entry["file"],
                        "path": entry["path"],
                        "rel": rel,
                        "chunk": i,
                        "content": piece,
                    })
            self._repository = repository
            self._files = sorted(rels)
            self._idf = self._compute_idf(docs)
            for doc in docs:
                doc["vector"] = self._vectorize(doc["content"], doc["path"])
            self._documents = docs
            self._save()
            return len(docs)

    def first_document(self) -> Optional[dict]:
        with self._lock:
            return dict(self._documents[0]) if self._documents else None

    # ------------------------------------------------------------------ #
    # Explorer support
    # ------------------------------------------------------------------ #

    def file_tree(self) -> List[dict]:
        with self._lock:
            if not self.ensure_loaded():
                return []
            return _build_tree(self._files)

    def file_count(self) -> int:
        with self._lock:
            self.ensure_loaded()
            return len(self._files)

    def file_chunks(self, rel_path: str, limit: int = 3) -> List[dict]:
        """Return indexed chunks for one repository-relative path."""
        with self._lock:
            if not self.ensure_loaded():
                return []
            limit = max(1, min(limit, MAX_PREVIEW_CHUNKS))
            rel = rel_path.strip().strip("/")
            out: List[dict] = []
            for doc in self._documents:
                doc_rel = doc.get("rel") or strip_repo_prefix(doc.get("path", ""), self._repository)
                if doc_rel == rel:
                    out.append({
                        "file": doc.get("file", ""),
                        "path": doc.get("path", ""),
                        "chunk": doc.get("chunk", 0),
                        "content": doc.get("content", ""),
                    })
                    if len(out) >= limit:
                        break
            return out

    # ------------------------------------------------------------------ #
    # Persistence (atomic write)
    # ------------------------------------------------------------------ #

    def _save(self) -> None:
        payload = {
            "version": INDEX_VERSION,
            "repository": self._repository,
            "idf": self._idf,
            "files": self._files,
            "documents": self._documents,
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.index_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
            os.replace(tmp_name, self.index_path)
        except OSError:
            logger.exception("Failed to persist index")
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------ #
    # Vectorisation
    # ------------------------------------------------------------------ #

    def _compute_idf(self, docs: List[dict]) -> Dict[str, float]:
        df: Counter = Counter()
        for doc in docs:
            df.update(set(tokenize(doc["content"] + " " + doc["path"])))
        total = max(len(docs), 1)
        return {t: math.log((1 + total) / (1 + c)) + 1.0 for t, c in df.items()}

    def _vectorize(self, content: str, path: str = "") -> Dict[str, float]:
        counts = Counter(tokenize(content + " " + path))
        if not counts:
            return {}
        vec = {t: (1 + math.log(tf)) * self._idf.get(t, 1.0)
               for t, tf in counts.items()}
        norm = math.sqrt(sum(w * w for w in vec.values())) or 1.0
        return {t: w / norm for t, w in vec.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return sum(w * b.get(t, 0.0) for t, w in a.items())  # unit vectors

    # ------------------------------------------------------------------ #
    # Hybrid search
    # ------------------------------------------------------------------ #

    def _query_terms(self, question: str) -> List[str]:
        terms = []
        for raw in _WORD_RE.findall(question):
            low = raw.lower()
            if len(low) < 2 or low in STOPWORDS:
                continue
            terms.append(SYNONYMS.get(low, low))
        return sorted(set(terms), key=len, reverse=True)

    def _exact_score(self, content_lower: str, path_lower: str, terms: List[str]) -> float:
        if not terms:
            return 0.0
        hits = 0.0
        for term in terms:
            if term in content_lower or term in path_lower:
                hits += 1.0
            elif _stem(term) in content_lower or _stem(term) in path_lower:
                hits += 0.5
        return hits / len(terms)

    def search(self, question: str, top_k: int = TOP_K) -> List[dict]:
        if not question or not question.strip():
            return []
        with self._lock:
            if not self.ensure_loaded():
                return []
            query_vec = self._vectorize(question)
            terms = self._query_terms(question)
            scored = []
            for doc in self._documents:
                content_lower = doc.get("content", "").lower()
                path_lower = doc.get("path", "").lower()
                exact = self._exact_score(content_lower, path_lower, terms)
                cosine = self._cosine(query_vec, doc.get("vector", {}))
                score = 0.6 * exact + 0.4 * cosine
                if score > MIN_SCORE:
                    scored.append((score, doc))
            scored.sort(key=lambda pair: pair[0], reverse=True)
            return [
                {
                    "file": doc.get("file", ""),
                    "path": doc.get("path", ""),
                    "rel": doc.get("rel") or strip_repo_prefix(doc.get("path", ""), self._repository),
                    "chunk": doc.get("chunk", 0),
                    "score": round(score, 3),
                    "content": doc.get("content", ""),
                }
                for score, doc in scored[:top_k]
            ]
