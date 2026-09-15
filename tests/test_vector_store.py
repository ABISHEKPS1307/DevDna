from backend.ai.vector_store import DevDNAVectorStore
from tests.conftest import sample_files


def test_build_counts_and_search(tmp_path):
    store = DevDNAVectorStore(tmp_path / "index.json")
    n = store.build("APIFlow", sample_files())
    assert n > 0
    results = store.search("Where is upload implemented?")
    assert results and results[0]["file"] == "main.py"
    assert results[0]["rel"] == "main.py"


def test_persistence_roundtrip(tmp_path):
    index = tmp_path / "index.json"
    store = DevDNAVectorStore(index)
    store.build("APIFlow", sample_files())
    reloaded = DevDNAVectorStore(index)
    assert reloaded.ensure_loaded()
    assert reloaded.repository == "APIFlow"
    assert reloaded.search("Where is upload implemented?")


def test_search_on_missing_index_returns_empty(tmp_path):
    store = DevDNAVectorStore(tmp_path / "missing.json")
    assert store.search("anything") == []


def test_irrelevant_query_returns_empty(tmp_path):
    store = DevDNAVectorStore(tmp_path / "index.json")
    store.build("APIFlow", sample_files())
    assert store.search("zzzqqqxxx") == []


def test_file_tree_and_chunks(tmp_path):
    store = DevDNAVectorStore(tmp_path / "index.json")
    store.build("APIFlow", sample_files())
    tree = store.file_tree()
    names = {node["name"]: node for node in tree}
    assert "main.py" in names
    assert names["templates"]["type"] == "dir"
    assert names["templates"]["children"][0]["path"] == "templates/index.html"
    chunks = store.file_chunks("main.py")
    assert chunks and chunks[0]["chunk"] == 1
    assert store.file_chunks("does_not_exist.py") == []
