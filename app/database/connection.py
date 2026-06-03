from __future__ import annotations

import os
import sqlite3
import logging
import json
from pathlib import Path
from datasets import load_dataset

logger = logging.getLogger(__name__)

DB_DIR = Path("database")
DB_PATH = DB_DIR / "beaver_dw.db"


TYPE_MAP = {
    "string": "TEXT",
    "int64": "INTEGER",
    "double": "REAL",
    "float64": "REAL",
    "bool": "INTEGER",
    "boolean": "INTEGER",
    "timestamp": "TEXT",
    "date": "TEXT"
}

def get_sqlite_type(huggingface_type: str) -> str:
    """Map HuggingFace schema types to SQLite types."""
    dtype = huggingface_type.lower()
    for key, val in TYPE_MAP.items():
        if key in dtype:
            return val
    return "TEXT"

def create_and_seed_db() -> str:
    """
    Creates the SQLite database at database/beaver_dw.db if it doesn't exist,
    defining tables dynamically from beaverbench/beaver-table columns, and seeding them.
    """
    DB_DIR.mkdir(parents=True, exist_ok=True)
    

    if DB_PATH.exists() and DB_PATH.stat().st_size > 0:
        logger.info(f"Database already exists at {DB_PATH}. Skipping seeding.")
        return str(DB_PATH)

    logger.info(f"Creating and seeding persistent SQLite database at {DB_PATH}...")
    
    
    try:
        table_ds = load_dataset("beaverbench/beaver-table", split="dw")
    except Exception as e:
        logger.error(f"Failed to load beaver-table for database seeding: {e}")
        raise RuntimeError(f"Could not load beaver-table for database creation: {e}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        for row in table_ds:
            table_name = row["table_name"]
            column_names_raw = row.get("column_names", "")
            column_types_raw = row.get("column_types", "")
            
            try:
                col_names = json.loads(column_names_raw)
            except Exception:
                col_names = [c.strip(" []\"'") for c in column_names_raw.split(",") if c.strip()]

            try:
                col_types = json.loads(column_types_raw)
            except Exception:
                col_types = [c.strip(" []\"'") for c in column_types_raw.split(",") if c.strip()]

            if len(col_types) < len(col_names):
                col_types += ["unknown"] * (len(col_names) - len(col_types))

            # Generate column definitions
            col_defs = []
            for name, dtype in zip(col_names, col_types):
                sqlite_type = get_sqlite_type(dtype)
                col_defs.append(f'"{name}" {sqlite_type}')

            # Create table DDL
            ddl = f'CREATE TABLE IF NOT EXISTS "{table_name}" (\n    ' + ",\n    ".join(col_defs) + "\n);"
            cursor.execute(ddl)
            logger.debug(f"Created table {table_name} with DDL:\n{ddl}")

        conn.commit()
        logger.info(f"Successfully created and initialized all 97 tables in {DB_PATH}.")
    except Exception as e:
        logger.error(f"Error seeding SQLite tables: {e}")
        conn.rollback()
        raise e
    finally:
        conn.close()

    return str(DB_PATH)

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the persistent SQLite database."""
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        create_and_seed_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
