import os
import shutil


def test_retriever_with_fake_collection(tmp_path, monkeypatch):
    # Build a tiny chroma collection with fake embeddings and docs
    try:
        import chromadb
    except Exception:
        import pytest
        pytest.skip("chromadb not installed")

    class FakeEmbedding:
        def __call__(self, input):
            # Deterministic small vectors
            if isinstance(input, list):
                return [[float((hash(x) % 100) / 100.0) for _ in range(8)] for x in input]
            return [[float((hash(input) % 100) / 100.0) for _ in range(8)]]
        def embed_query(self, input):
            return self.__call__(input)
        def embed_documents(self, input):
            return self.__call__(input)
        def name(self):
            return "fake"
        @property
        def is_legacy(self):
            return False

    persist_dir = str(tmp_path / "rag_index")
    client = chromadb.PersistentClient(path=persist_dir)
    coll = client.get_or_create_collection(name="schema_index", embedding_function=FakeEmbedding())

    docs = [
        ("table::orders", "TABLE orders: columns[id INTEGER, customer_id INTEGER, total NUMERIC]", {"type": "table", "table": "orders"}),
        ("table::customers", "TABLE customers: columns[id INTEGER, name TEXT]", {"type": "table", "table": "customers"}),
        ("column::orders.total", "COLUMN orders.total type NUMERIC", {"type": "column", "table": "orders", "column": "total"}),
        ("column::orders.customer_id", "COLUMN orders.customer_id type INTEGER", {"type": "column", "table": "orders", "column": "customer_id"}),
        ("column::customers.name", "COLUMN customers.name type TEXT", {"type": "column", "table": "customers", "column": "name"}),
    ]
    coll.upsert(ids=[d[0] for d in docs], documents=[d[1] for d in docs], metadatas=[d[2] for d in docs])

    # Now run retriever
    from rag.retriever import SchemaRetriever, build_context_block
    retr = SchemaRetriever(persist_dir=persist_dir, embedding_fn=FakeEmbedding())
    result = retr.retrieve_relevant_schema("total by customer")

    assert isinstance(result, dict)
    assert "tables" in result and "columns" in result

    ctx = build_context_block(result)
    assert isinstance(ctx, str)
