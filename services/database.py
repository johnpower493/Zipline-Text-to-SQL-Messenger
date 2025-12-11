"""
Database service for SQLite operations and schema management.
"""
import sqlite3
import json
import os
import yaml
from typing import Optional, List, Dict, Any
from config.settings import config


class DatabaseService:
    """Handles all database operations and schema management."""
    
    @staticmethod
    def load_schema_from_yaml(path: str) -> Optional[str]:
        """Load database schema from YAML file."""
        if not path or not os.path.exists(path):
            return None
        
        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f)
                return cfg.get("database_schema")
        except Exception:
            return None
    
    @staticmethod
    def load_schema_from_sqlite(db_path: str) -> str:
        """
        Build a compact, LLM-friendly schema string from SQLite PRAGMAs.
        
        Args:
            db_path: Path to SQLite database
            
        Returns:
            Schema string describing tables and relationships
        """
        if not os.path.exists(db_path):
            return "Database file not found."
        
        try:
            # Use parameterized connection for safety
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            
            # Get all tables (excluding SQLite system tables)
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
            tables = [row["name"] for row in cur.fetchall()]
            
            schema_lines = []
            
            for table_name in tables:
                # Get column information
                # Using ? placeholder to prevent SQL injection
                cur.execute("SELECT * FROM pragma_table_info(?)", (table_name,))
                columns = cur.fetchall()
                
                col_descriptions = []
                for col in columns:
                    name = col["name"]
                    col_type = col["type"]
                    pk_suffix = " PK" if col["pk"] else ""
                    col_descriptions.append(f"{name} {col_type}{pk_suffix}")
                
                # Get foreign key information
                cur.execute("SELECT * FROM pragma_foreign_key_list(?)", (table_name,))
                foreign_keys = cur.fetchall()
                
                fk_descriptions = []
                for fk in foreign_keys:
                    fk_descriptions.append(f"{fk['from']} -> {fk['table']}.{fk['to']}")
                
                # Build table description
                table_desc = f"- {table_name}(" + ", ".join(col_descriptions) + ")"
                if fk_descriptions:
                    table_desc += " FKs[" + "; ".join(fk_descriptions) + "]"
                
                schema_lines.append(table_desc)
            
            con.close()
            return "\n".join(schema_lines)
            
        except Exception as e:
            return f"Error reading database schema: {str(e)}"
    
    @staticmethod
    def get_database_schema() -> str:
        """Get the database schema from YAML or auto-generate from SQLite."""
        schema = DatabaseService.load_schema_from_yaml(config.SCHEMA_YAML_PATH)
        if schema:
            return schema
        return DatabaseService.load_schema_from_sqlite(config.SQLITE_PATH)
    
    @staticmethod
    def explain_query(sql_query: str) -> Dict[str, Any]:
        """
        Preflight validation of SQL using EXPLAIN QUERY PLAN.
        Returns {'ok': True} or {'error': '...'}
        """
        try:
            con = sqlite3.connect(f"file:{config.SQLITE_PATH}?mode=ro", uri=True)
            cur = con.cursor()
            cur.execute(f"EXPLAIN QUERY PLAN {sql_query}")
            _ = cur.fetchall()
            con.close()
            return {"ok": True}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    def execute_query(sql_query: str) -> Dict[str, Any]:
        """
        Execute a SQL query and return results as JSON.
        
        Args:
            sql_query: The SQL query to execute
            
        Returns:
            Dictionary with either 'data' (list of dicts) or 'error' key
        """
        try:
            # Open in read-only mode for safety
            con = sqlite3.connect(f"file:{config.SQLITE_PATH}?mode=ro", uri=True)
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            
            cur.execute(sql_query)
            rows = cur.fetchall()
            result = [dict(row) for row in rows]
            
            con.close()
            return {"data": result}
            
        except Exception as e:
            return {"error": str(e)}