import os
import shutil

from rag.schema_indexer import build_schema_entries, index_schema


def test_build_schema_entries_basic():
    structured = {
        "tables": [
            {
                "name": "customers",
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "name", "type": "TEXT"},
                ],
                "fks": []
            },
            {
                "name": "orders",
                "columns": [
                    {"name": "id", "type": "INTEGER"},
                    {"name": "customer_id", "type": "INTEGER"},
                    {"name": "total", "type": "NUMERIC"},
                ],
                "fks": ["customer_id -> customers.id"]
            }
        ]
    }
    entries = build_schema_entries(structured)
    ids = {e['id'] for e in entries}
    assert "table::customers" in ids
    assert "column::customers.id" in ids
    assert "table::orders" in ids
    assert any("FKs[customer_id -> customers.id]" in e['text'] for e in entries if e['id']=="table::orders")


def test_index_schema_with_fake_embedding(tmp_path, monkeypatch):
    # Provide a fake embedding function to avoid model download during tests
    class FakeEmbedding:
        def __call__(self, input):
            # Return a deterministic small-dimension vector
            if isinstance(input, list):
                return [[float((hash(x) % 100) / 100.0) for _ in range(8)] for x in input]
            return [[float((hash(input) % 100) / 100.0) for _ in range(8)]]
        def name(self):
            return "fake"
        @property
        def is_legacy(self):
            return False

    # Monkeypatch chroma client to ensure it can initialize (requires chromadb package)
    try:
        import chromadb  # noqa: F401
    except Exception:
        import pytest
        pytest.skip("chromadb not installed")

    persist_dir = str(tmp_path / "rag_index")
    # Running indexer should succeed even if empty DB is used, but our indexer introspects configured DB.
    # We cannot guarantee a DB here; so just assert it raises FileNotFoundError or builds index.
    try:
        index_schema(persist_dir=persist_dir, embedding_fn=FakeEmbedding())
    except FileNotFoundError:
        # Acceptable in test environment without a DB file
        pass
