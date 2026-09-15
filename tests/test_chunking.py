from backend.ai.vector_store import chunk_text, tokenize


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_small_text_single_chunk():
    assert chunk_text("x = 1") == ["x = 1"]


def test_large_text_multiple_chunks_with_overlap():
    text = "\n".join(f"line_{i} = {i}" for i in range(120))
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) >= 3
    assert all(len(c) <= 500 for c in chunks)
    assert chunks[1] != chunks[0]


def test_tokenizer_splits_identifiers():
    assert tokenize("Jinja2Templates") == ["jinja2", "template"]
    assert tokenize("upload_excel") == ["upload", "excel"]
    assert tokenize("the") == []          # stopword
    assert tokenize("routes") == ["endpoint"]  # synonym + stem
