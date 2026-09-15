"""DevDNA – AI-Powered Codebase Knowledge Assistant (FastAPI backend)."""

from __future__ import annotations

import logging
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from backend.ai.rag import rag_service
from backend.ai.vector_store import DevDNAVectorStore
from backend.utils.code_reader import (
    MAX_UPLOAD_BYTES, InvalidZipError, detect_language, derive_repo_name,
    find_repo_root, safe_extract, scan_repository,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("devdna")

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
REPOSITORIES_DIR = BASE_DIR / "repositories"
INDEX_PATH = REPOSITORIES_DIR / ".devdna_index.json"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
REPOSITORIES_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DevDNA", version="1.1.0",
              description="AI-Powered Codebase Knowledge Assistant")
templates = Jinja2Templates(directory=str(FRONTEND_DIR))
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
vector_store = DevDNAVectorStore(INDEX_PATH)


# --------------------------------------------------------------------------- #
# Pages & health
# --------------------------------------------------------------------------- #

@app.get("/")
async def homepage(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    return {"status": "healthy", "ai_available": rag_service.available()}


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #

def _cleanup(zip_path: Path, dest_root: Path) -> None:
    try:
        zip_path.unlink(missing_ok=True)
    except OSError:
        pass
    shutil.rmtree(dest_root, ignore_errors=True)


@app.post("/upload-repository")
async def upload_repository(file: UploadFile = File(...)):
    filename = file.filename or ""
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400,
                            detail="Please upload a .zip archive (GitHub 'Download ZIP' works great).")

    zip_path = UPLOADS_DIR / f"{uuid.uuid4().hex[:8]}-{Path(filename).name}"
    size = 0
    try:
        with zip_path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=400,
                                        detail="ZIP file is too large (100 MB max).")
                fh.write(chunk)
    except HTTPException:
        _cleanup(zip_path, REPOSITORIES_DIR)
        raise

    dest_root = REPOSITORIES_DIR / uuid.uuid4().hex[:12]
    try:
        safe_extract(zip_path, dest_root)
    except zipfile.BadZipFile:
        _cleanup(zip_path, dest_root)
        raise HTTPException(status_code=400, detail="That file is not a valid ZIP archive.")
    except InvalidZipError as exc:
        _cleanup(zip_path, dest_root)
        raise HTTPException(status_code=400, detail=str(exc))

    repo_root = find_repo_root(dest_root)
    repo_name = derive_repo_name(Path(filename).stem, repo_root, dest_root)

    scan = scan_repository(repo_root)
    if scan["total_code_files"] == 0:
        _cleanup(zip_path, dest_root)
        raise HTTPException(
            status_code=400,
            detail="No supported code files were found (Python, JS, TS, JSX, TSX, Java, HTML, CSS).")

    files = [{"file": Path(rel).name, "path": f"{repo_name}/{rel}", "rel": rel,
              "content": content} for rel, content in scan["code_texts"]]
    try:
        chunks_created = vector_store.build(repo_name, files)
    except Exception:
        logger.exception("Index build failed")
        _cleanup(zip_path, dest_root)
        raise HTTPException(status_code=500,
                            detail="Failed to index the repository. Please try again.")

    logger.info("Indexed %s: %d files, %d chunks", repo_name,
                scan["total_code_files"], chunks_created)
    first = vector_store.first_document() or {}
    return {
        "repository": repo_name,
        "total_code_files": scan["total_code_files"],
        "framework": scan["framework"],
        "endpoints": scan["endpoints"],
        "languages": scan["languages"],
        "chunks_created": chunks_created,
        "detected_files": scan["detected_files"],
        "sample_chunk": first.get("content", ""),
    }


# --------------------------------------------------------------------------- #
# Repository explorer
# --------------------------------------------------------------------------- #

@app.get("/repository/files")
async def repository_files():
    if not vector_store.ensure_loaded():
        raise HTTPException(status_code=404,
                            detail="No repository indexed yet. Upload a ZIP first.")
    return {
        "repository": vector_store.repository,
        "total_files": vector_store.file_count(),
        "tree": vector_store.file_tree(),
    }


@app.get("/repository/preview")
async def repository_preview(path: str, limit: int = 3):
    clean = (path or "").strip().replace("\\", "/").lstrip("/")
    if not clean or ".." in clean.split("/"):
        raise HTTPException(status_code=400, detail="Invalid file path.")
    chunks = vector_store.file_chunks(clean, limit=limit)
    if not chunks:
        if not vector_store.ensure_loaded():
            raise HTTPException(status_code=404,
                                detail="No repository indexed yet. Upload a ZIP first.")
        raise HTTPException(status_code=404,
                            detail="File not found in the indexed repository.")
    language = detect_language(Path(clean)) or "Code"
    return {"path": clean, "language": language, "chunks": chunks}


# --------------------------------------------------------------------------- #
# Ask (hybrid local search + optional OpenAI RAG)
# --------------------------------------------------------------------------- #

class AskRequest(BaseModel):
    question: str
    mode: Literal["local", "ai"] = "local"


@app.post("/ask")
async def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please type a question about your repository.")

    if not vector_store.ensure_loaded():
        raise HTTPException(status_code=404,
                            detail="No repository indexed yet. Upload a ZIP first.")

    top_k = 4 if payload.mode == "ai" else 3
    matches = vector_store.search(question, top_k=top_k)
    if not matches:
        return {
            "answer": ("I couldn't find anything in the indexed repository matching "
                       "that question. Try naming a function, class, route or keyword "
                       "that actually exists in the code."),
            "matches": [],
            "mode_used": "local",
            "notice": None,
        }

    mode_used = "local"
    notice: Optional[str] = None

    if payload.mode == "ai":
        if rag_service.available():
            try:
                # Sync OpenAI SDK call -> run in a worker thread.
                answer = await run_in_threadpool(rag_service.generate_answer, question, matches)
                mode_used = "ai"
            except Exception:
                logger.exception("RAG generation failed; falling back to local answer")
                answer = generate_answer(question, matches)
                notice = "AI request failed — showing local search results instead."
        else:
            answer = generate_answer(question, matches)
            notice = ("AI mode unavailable — set OPENAI_API_KEY (and install the openai "
                      "package) to enable it. Showing local search results.")
    else:
        answer = generate_answer(question, matches)

    return {"answer": answer, "matches": matches,
            "mode_used": mode_used, "notice": notice}


# --------------------------------------------------------------------------- #
# Local template-based answer generation (evidence-grounded)
# --------------------------------------------------------------------------- #

_ROUTE_RE = re.compile(r"@\s*\w[\w.]*\.(get|post|put|delete|patch)\s*\(\s*[\"']([^\"']+)[\"']")
_DEF_RE = re.compile(r"\bdef\s+([A-Za-z_]\w*)")
_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_]\w*)")

_SUBJECT_STOP = {
    "where", "what", "how", "which", "explain", "does", "this", "that", "the",
    "is", "are", "in", "and", "for", "with", "implemented", "implement",
    "initialize", "initialized", "initialise", "initialised", "used", "use",
    "find", "show", "tell", "about", "code", "file", "work", "works", "working",
}


def _subject(question: str) -> str:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", question)
    candidates = [w for w in words if w.lower() not in _SUBJECT_STOP]
    return max(candidates, key=len) if candidates else ""


def generate_answer(question: str, matches: List[dict]) -> str:
    top = matches[0]
    where = top.get("path") or top.get("file") or "the repository"
    content = top.get("content", "")
    subject = _subject(question)
    subject_text = subject or "This functionality"

    route = _ROUTE_RE.search(content)
    func = _DEF_RE.search(content)
    klass = _CLASS_RE.search(content)

    if route:
        opener = (f"{subject_text} is handled in {where} through the "
                  f"{route.group(1).upper()} {route.group(2)} endpoint")
    elif func:
        opener = f"{subject_text} lives in {where}, inside the `{func.group(1)}` function"
    elif klass:
        opener = f"{subject_text} is defined in {where}, in the `{klass.group(1)}` class"
    else:
        opener = (f"The most relevant code for {subject_text} is in {where} "
                  f"(chunk {top.get('chunk', 1)})")

    parts = [opener + "."]
    if func and route:
        parts.append(f"The route is implemented by the `{func.group(1)}` function.")

    if subject:
        for line in content.splitlines():
            stripped = line.strip()
            if subject.lower() in stripped.lower() and len(stripped) <= 120 and "`" not in stripped:
                parts.append(f"Key line: `{stripped}`")
                break

    seen, extras = {top.get("path", "")}, []
    for m in matches[1:]:
        p = m.get("path", "")
        if p and p not in seen:
            seen.add(p)
            extras.append(p)
    if extras:
        parts.append("Related code also appears in " + ", ".join(extras[:2]) + ".")

    parts.append("The matching code is shown below.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Error handling – never leak tracebacks
# --------------------------------------------------------------------------- #

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error. Please try again."})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
