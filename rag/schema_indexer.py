"""
Schema indexer for building a local vector index over database schemas.
- Uses SQLite introspection (via DatabaseService) to extract tables and columns
- Embeds using sentence-transformers (BAAI/bge-small-en-v1.5) by default
- Stores vectors in a local ChromaDB collection

CLI:
    python -m rag.schema_indexer --config path/to/schema.yaml

The --config flag is optional; current implementation uses the configured
SQLite database for structured schema extraction and ignores custom YAML
for embedding granularity. You can extend build_schema_entries to merge
YAML descriptions if available.
"""
from typing import List, Dict, Any, Optional
import os
import argparse
import logging

from config.settings import config
from services.database import DatabaseService

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None


def build_schema_entries(structured_schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert structured schema into indexable entries for tables and columns.
    structured_schema format expected:
        {
          "tables": [
              {"name": str, "columns": [{"name": str, "type": str}], "fks": ["a -> b.c", ...]},
              ...
          ]
        }
    Returns a list of entries with fields: id, text, metadata
    """
    entries: List[Dict[str, Any]] = []
    for t in structured_schema.get("tables", []):
        table_name = t["name"]
        col_list = ", ".join([f"{c['name']} {c.get('type','')}".strip() for c in t.get("columns", [])])
        fk_list = "; ".join(t.get("fks", []))
        table_text = f"TABLE {table_name}: columns[{col_list}]"
        if fk_list:
            table_text += f" FKs[{fk_list}]"
        entries.append({
            "id": f"table::{table_name}",
            "text": table_text,
            "metadata": {"type": "table", "table": table_name}
        })
        # Column-level entries
        for c in t.get("columns", []):
            col_text = f"COLUMN {table_name}.{c['name']} type {c.get('type','')}"
            entries.append({
                "id": f"column::{table_name}.{c['name']}",
                "text": col_text,
                "metadata": {"type": "column", "table": table_name, "column": c['name']}
            })
    return entries


def _wrap_embedding_fn(embedding_fn):
    """Wraps a bare embedding callable with a minimal adapter exposing name()."""
    if embedding_fn is None:
        return None
    # If it already has a name() method, pass through
    if hasattr(embedding_fn, "name") and callable(getattr(embedding_fn, "name")):
        return embedding_fn

    class _Adapter:
        def __init__(self, base):
            self._base = base
        def __call__(self, input):
            return self._base(input)
        def name(self):
            return "custom"
        @property
        def is_legacy(self):  # new chroma config checks
            return False
        def embed_query(self, input):
            return self.__call__(input)
        def embed_documents(self, input):
            return self.__call__(input)
    return _Adapter(embedding_fn)


def get_default_embedding_function():
    """Return the default sentence-transformers embedding function for Chroma."""
    if embedding_functions is None:
        raise RuntimeError("chromadb not installed")
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5"
    )


def index_schema(persist_dir: Optional[str] = None, embedding_fn=None) -> str:
    """Build or refresh the schema vector index.
    Returns the persist directory path used.
    """
    if chromadb is None:
        raise RuntimeError("chromadb not installed. Please install dependencies.")

    persist_dir = persist_dir or config.RAG_PERSIST_DIR
    os.makedirs(persist_dir, exist_ok=True)

    client = chromadb.PersistentClient(path=persist_dir)

    # Use a single collection for schema
    ef = _wrap_embedding_fn(embedding_fn) or get_default_embedding_function()
    collection = client.get_or_create_collection(
        name="schema_index",
        metadata={"hnsw:space": "cosine"},
        embedding_function=ef,
    )

    # Build structured schema using DatabaseService
    # Extend DatabaseService to offer structured metadata if available
    structured_schema = _get_structured_schema()
    entries = build_schema_entries(structured_schema)

    if not entries:
        logger.warning("No schema entries found to index")
        return persist_dir

    # Upsert into Chroma
    ids = [e["id"] for e in entries]
    documents = [e["text"] for e in entries]
    metadatas = [e["metadata"] for e in entries]

    # Chroma upsert in batches to avoid large payloads
    BATCH = 128
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i+BATCH],
            documents=documents[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

    logger.info(f"Indexed {len(ids)} schema entries into {persist_dir}")
    return persist_dir


def _get_structured_schema() -> Dict[str, Any]:
    """Create a structured schema by introspecting SQLite via DatabaseService."""
    # Reuse DatabaseService logic to introspect
    import sqlite3
    import os as _os

    db_path = config.SQLITE_PATH
    if not _os.path.exists(db_path):
        raise FileNotFoundError(f"SQLite database not found at {db_path}")

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
    tables = [row["name"] for row in cur.fetchall()]

    out = {"tables": []}
    for table_name in tables:
        cur.execute("SELECT * FROM pragma_table_info(?)", (table_name,))
        columns = cur.fetchall()
        col_list = [{"name": col["name"], "type": col["type"], "pk": int(col["pk"]) == 1} for col in columns]

        cur.execute("SELECT * FROM pragma_foreign_key_list(?)", (table_name,))
        fks = cur.fetchall()
        fk_desc = [f"{fk['from']} -> {fk['table']}.{fk['to']}" for fk in fks]

        out["tables"].append({"name": table_name, "columns": col_list, "fks": fk_desc})

    con.close()
    return out


def main():
    parser = argparse.ArgumentParser(description="Build/refresh schema vector index")
    parser.add_argument("--config", help="Optional schema config path (currently unused)", default=None)
    parser.add_argument("--persist_dir", help="Directory to persist vector DB", default=None)
    args = parser.parse_args()

    try:
        used_dir = index_schema(persist_dir=args.persist_dir)
        print(f"✅ Schema index built at: {used_dir}")
    except Exception as e:
        print(f"❌ Failed to build schema index: {e}")
        raise

if __name__ == "__main__":
    main()
