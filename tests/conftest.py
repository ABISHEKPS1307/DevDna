import io
import zipfile

import pytest
from fastapi.testclient import TestClient

import backend.main as main
from backend.ai.vector_store import DevDNAVectorStore

MAIN_PY = '''\
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import pandas as pd
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/")
async def homepage(request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/upload-excel")
async def upload_excel(file: UploadFile):
    df = pd.read_excel(file.file)
    df.to_excel("uploads/output.xlsx")
    return {"rows": len(df)}

@app.get("/download-excel")
async def download_excel():
    return FileResponse("uploads/output.xlsx", filename="report.xlsx")
'''

INDEX_HTML = "<html><body><h1>APIFlow</h1><script>fetch('/download-excel')</script></body></html>"


def make_repo_zip_bytes(top_dir: str = "APIFlow-main") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{top_dir}/main.py", MAIN_PY)
        zf.writestr(f"{top_dir}/templates/index.html", INDEX_HTML)
        zf.writestr(f"{top_dir}/node_modules/junk.js", "console.log('ignored');")
        zf.writestr(f"{top_dir}/README.md", "# APIFlow")
    return buf.getvalue()


def extract_to(tmp_path, top_dir="APIFlow-main"):
    """Helper used by unit tests to get an extracted repo on disk."""
    from backend.utils.code_reader import safe_extract, find_repo_root
    zip_path = tmp_path / "repo.zip"
    zip_path.write_bytes(make_repo_zip_bytes(top_dir))
    dest = tmp_path / "extracted"
    safe_extract(zip_path, dest)
    return find_repo_root(dest), dest


def sample_files(repo="APIFlow", main_py=MAIN_PY, index_html=INDEX_HTML):
    """Vector-store entries matching the canonical path contract."""
    return [
        {"file": "main.py", "path": f"{repo}/main.py", "rel": "main.py", "content": main_py},
        {"file": "index.html", "path": f"{repo}/templates/index.html",
         "rel": "templates/index.html", "content": index_html},
    ]


@pytest.fixture()
def sample_zip_bytes():
    return make_repo_zip_bytes()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """API client isolated from the real project directories."""
    uploads = tmp_path / "uploads"
    repos = tmp_path / "repositories"
    index = repos / ".devdna_index.json"
    monkeypatch.setattr(main, "UPLOADS_DIR", uploads)
    monkeypatch.setattr(main, "REPOSITORIES_DIR", repos)
    monkeypatch.setattr(main, "INDEX_PATH", index)
    monkeypatch.setattr(main, "vector_store", DevDNAVectorStore(index))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield TestClient(main.app)


@pytest.fixture()
def uploaded_client(client, sample_zip_bytes):
    res = client.post("/upload-repository",
                      files={"file": ("APIFlow-main.zip", sample_zip_bytes, "application/zip")})
    assert res.status_code == 200
    return client
