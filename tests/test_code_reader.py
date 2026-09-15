import zipfile

from backend.utils.code_reader import (
    derive_repo_name, extract_to, find_repo_root, scan_repository,  # noqa: F401 (extract_to from conftest)
)
from tests.conftest import extract_to  # canonical import


def test_safe_extract_creates_files(tmp_path):
    root, _ = extract_to(tmp_path)
    assert (root / "main.py").exists()
    assert (root / "templates" / "index.html").exists()


def test_zip_slip_member_is_rejected(tmp_path):
    import io
    from backend.utils.code_reader import safe_extract
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.py", "print('pwned')")
        zf.writestr("ok/main.py", "x = 1")
    dest = tmp_path / "dest"
    safe_extract(tmp_path / "t.zip" if False else _write_zip(buf, tmp_path), dest)
    assert not (tmp_path.parent / "evil.py").exists()   # nothing escaped dest
    assert not (tmp_path / "evil.py").exists()
    assert (dest / "ok" / "main.py").exists()           # safe members still extract


def _write_zip(buf, tmp_path):
    p = tmp_path / "hostile.zip"
    p.write_bytes(buf.getvalue())
    return p


def test_find_repo_root_flattens_github_layout(tmp_path):
    _, dest = extract_to(tmp_path)
    root = find_repo_root(dest)
    assert root.name == "APIFlow-main"


def test_derive_repo_name_strips_branch_suffix():
    assert derive_repo_name("APIFlow-main", None, None) == "APIFlow"
    assert derive_repo_name("myrepo-master", None, None) == "myrepo"
    assert derive_repo_name("plain", None, None) == "plain"


def test_scan_repository_detection(tmp_path):
    root, _ = extract_to(tmp_path)
    scan = scan_repository(root)
    assert scan["total_code_files"] == 2          # node_modules + README ignored
    assert scan["languages"] == {"Python": 1, "HTML": 1}
    assert scan["framework"] == "FastAPI"
    assert scan["endpoints"] == 3
    assert "templates/index.html" in scan["detected_files"]
