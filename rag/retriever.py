"""
Retriever for pulling relevant schema tables/columns using local Chroma index.
"""
from typing import Dict, Any, List, Optional, Tuple
import os
import logging

from config.settings import config

logger = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None


def retrieve_relevant_schema(question: str, persist_dir: Optional[str] = None, embedding_fn=None) -> Dict[str, Any]:
    """Convenience function to retrieve relevant schema context.
    Returns a dict: {"tables": [...], "columns": {table: [cols]}}
    """
    sr = SchemaRetriever(persist_dir=persist_dir, embedding_fn=embedding_fn)
    return sr.retrieve_relevant_schema(question)


def _wrap_embedding_fn(embedding_fn):
    if embedding_fn is None:
        return None
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
        def is_legacy(self):
            return False
        # Add methods used by recent Chroma versions
        def embed_query(self, input):
            return self.__call__(input)
        def embed_documents(self, input):
            return self.__call__(input)
    return _Adapter(embedding_fn)


class SchemaRetriever:
    def __init__(self, persist_dir: Optional[str] = None, embedding_fn=None):
        if chromadb is None:
            raise RuntimeError("chromadb not installed")
        self.persist_dir = persist_dir or config.RAG_PERSIST_DIR
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        ef = _wrap_embedding_fn(embedding_fn) or self._default_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name="schema_index",
            metadata={"hnsw:space": "cosine"},
            embedding_function=ef,
        )
        # Ensure the runtime embedding function supports embed_query even if collection existed already
        try:
            self.collection._embedding_function = ef  # type: ignore[attr-defined]
        except Exception:
            pass
        self.top_k_tables = max(1, config.RAG_TOP_K_TABLES)
        self.top_n_cols = max(1, config.RAG_TOP_N_COLUMNS)

    def _default_embedding_function(self):
        if embedding_functions is None:
            raise RuntimeError("chromadb not installed")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def retrieve_relevant_schema(self, question: str) -> Dict[str, Any]:
        """Return a structured result: {tables: [table_names], columns: {table: [col_names]}}
        """
        if not question:
            return {"tables": [], "columns": {}}

        # Search columns first for precision
        col_results = self.collection.query(
            query_texts=[question],
            n_results=self.top_k_tables * self.top_n_cols,
            where={"type": "column"}
        )
        tables_from_cols: List[str] = []
        columns_map: Dict[str, List[str]] = {}

        for md in col_results.get("metadatas", [[]])[0]:
            table = md.get("table")
            col = md.get("column")
            if not table or not col:
                continue
            if table not in columns_map:
                columns_map[table] = []
            if col not in columns_map[table]:
                columns_map[table].append(col)
            if table not in tables_from_cols:
                tables_from_cols.append(table)

        # Search tables entries to complement
        table_results = self.collection.query(
            query_texts=[question],
            n_results=self.top_k_tables,
            where={"type": "table"}
        )
        tables_ranked: List[str] = tables_from_cols.copy()
        for md in table_results.get("metadatas", [[]])[0]:
            t = md.get("table")
            if t and t not in tables_ranked:
                tables_ranked.append(t)

        # Truncate to top_k_tables
        tables_ranked = tables_ranked[: self.top_k_tables]

        # For each selected table, limit columns to top_n
        for t in list(columns_map.keys()):
            if t not in tables_ranked:
                columns_map.pop(t, None)
        for t in tables_ranked:
            if t in columns_map:
                columns_map[t] = columns_map[t][: self.top_n_cols]

        return {"tables": tables_ranked, "columns": columns_map}


def build_context_block(retrieved: Dict[str, Any]) -> str:
    """Create a compact schema context string from retrieved tables/columns."""
    lines: List[str] = []
    for t in retrieved.get("tables", []):
        cols = retrieved.get("columns", {}).get(t, [])
        if cols:
            cols_str = ", ".join(cols)
            lines.append(f"- {t}({cols_str})")
        else:
            lines.append(f"- {t}")
    return "\n".join(lines) if lines else ""
