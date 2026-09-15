import io
import zipfile

import backend.main as main
from tests.conftest import make_repo_zip_bytes


def test_health(client):
    data = client.get("/health").json()
    assert data["status"] == "healthy"
    assert data["ai_available"] is False          # no OPENAI_API_KEY in fixture


def test_upload_success(client, sample_zip_bytes):
    res = client.post("/upload-repository",
                      files={"file": ("APIFlow-main.zip", sample_zip_bytes, "application/zip")})
    assert res.status_code == 200
    data = res.json()
    assert data["repository"] == "APIFlow"
    assert data["total_code_files"] == 2
    assert data["framework"] == "FastAPI"
    assert data["endpoints"] == 3
    assert data["languages"] == {"Python": 1, "HTML": 1}
    assert data["chunks_created"] > 0
    assert data["sample_chunk"]
    assert "templates/index.html" in data["detected_files"]


def test_upload_rejects_non_zip(client):
    res = client.post("/upload-repository",
                      files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 400


def test_upload_rejects_corrupt_zip(client):
    res = client.post("/upload-repository",
                      files={"file": ("bad.zip", b"this is not a zip", "application/zip")})
    assert res.status_code == 400
    assert "not a valid ZIP" in res.json()["detail"]


def test_upload_rejects_zip_without_code_files(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("only/README.md", "# nothing to index")
    res = client.post("/upload-repository",
                      files={"file": ("empty.zip", buf.getvalue(), "application/zip")})
    assert res.status_code == 400
    assert "No supported code files" in res.json()["detail"]


def test_ask_success(uploaded_client):
    res = uploaded_client.post("/ask", json={"question": "Where is Jinja2Templates initialized?"})
    assert res.status_code == 200
    data = res.json()
    assert data["matches"] and len(data["matches"]) <= 3
    assert "main.py" in data["answer"]
    assert data["mode_used"] == "local"
    assert data["notice"] is None


def test_ask_empty_results(uploaded_client, monkeypatch):
    monkeypatch.setattr(main.vector_store, "search", lambda q, top_k=3: [])
    res = uploaded_client.post("/ask", json={"question": "anything obscure"})
    assert res.status_code == 200
    data = res.json()
    assert data["matches"] == []
    assert "couldn't find" in data["answer"]


def test_ask_blank_question(uploaded_client):
    assert uploaded_client.post("/ask", json={"question": "   "}).status_code == 400


def test_ask_before_upload_returns_404(client):
    res = client.post("/ask", json={"question": "Where is upload implemented?"})
    assert res.status_code == 404


def test_ask_ai_mode_without_key_falls_back(uploaded_client):
    res = uploaded_client.post("/ask", json={"question": "Where is upload implemented?",
                                             "mode": "ai"})
    assert res.status_code == 200
    data = res.json()
    assert data["mode_used"] == "local"
    assert "unavailable" in data["notice"]


def test_ask_ai_mode_with_fake_rag(uploaded_client, monkeypatch):
    class FakeRAG:
        @staticmethod
        def available():
            return True

        @staticmethod
        def generate_answer(question, matches):
            return "AI GENERATED ANSWER"

    monkeypatch.setattr(main, "rag_service", FakeRAG())
    res = uploaded_client.post("/ask", json={"question": "Where is upload implemented?",
                                             "mode": "ai"})
    assert res.status_code == 200
    data = res.json()
    assert data["answer"] == "AI GENERATED ANSWER"
    assert data["mode_used"] == "ai"


def test_repository_files_endpoint(client, uploaded_client):
    res = client.get("/repository/files")
    assert res.status_code == 200
    data = res.json()
    assert data["repository"] == "APIFlow"
    paths = _flatten(data["tree"])
    assert "main.py" in paths
    assert "templates/index.html" in paths


def _flatten(nodes):
    out = []
    for n in nodes:
        if n["type"] == "file":
            out.append(n["path"])
        else:
            out.extend(_flatten(n.get("children", [])))
    return out


def test_repository_files_before_upload(client):
    assert client.get("/repository/files").status_code == 404


def test_repository_preview_valid(uploaded_client):
    res = uploaded_client.get("/repository/preview", params={"path": "main.py"})
    assert res.status_code == 200
    data = res.json()
    assert data["language"] == "Python"
    assert data["chunks"]


def test_repository_preview_traversal_rejected(uploaded_client):
    res = uploaded_client.get("/repository/preview", params={"path": "../secrets.env"})
    assert res.status_code == 400


def test_repository_preview_unknown_file(uploaded_client):
    assert uploaded_client.get("/repository/preview",
                               params={"path": "ghost.py"}).status_code == 404
