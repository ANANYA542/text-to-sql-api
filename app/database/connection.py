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

class VarianceAggregate:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            try:
                self.values.append(float(value))
            except (ValueError, TypeError):
                pass
    def finalize(self):
        n = len(self.values)
        if n <= 1:
            return 0.0 if n == 1 else None
        mean = sum(self.values) / n
        return sum((x - mean) ** 2 for x in self.values) / (n - 1)

class StddevAggregate:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            try:
                self.values.append(float(value))
            except (ValueError, TypeError):
                pass
    def finalize(self):
        n = len(self.values)
        if n <= 1:
            return 0.0 if n == 1 else None
        mean = sum(self.values) / n
        var = sum((x - mean) ** 2 for x in self.values) / (n - 1)
        import math
        return math.sqrt(var)

class VariancePopAggregate:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            try:
                self.values.append(float(value))
            except (ValueError, TypeError):
                pass
    def finalize(self):
        n = len(self.values)
        if n == 0:
            return None
        mean = sum(self.values) / n
        return sum((x - mean) ** 2 for x in self.values) / n

class StddevPopAggregate:
    def __init__(self):
        self.values = []
    def step(self, value):
        if value is not None:
            try:
                self.values.append(float(value))
            except (ValueError, TypeError):
                pass
    def finalize(self):
        n = len(self.values)
        if n == 0:
            return None
        mean = sum(self.values) / n
        var = sum((x - mean) ** 2 for x in self.values) / n
        import math
        return math.sqrt(var)

def register_custom_functions(conn: sqlite3.Connection):
    import math
    
    # Register scalar functions
    def sqlite_log(x):
        if x is None:
            return None
        try:
            val = float(x)
            if val <= 0:
                return None
            return math.log(val)
        except Exception:
            return None

    def sqlite_exp(x):
        if x is None:
            return None
        try:
            return math.exp(float(x))
        except Exception:
            return None

    def sqlite_ascii(x):
        if not x:
            return None
        try:
            s = str(x)
            return ord(s[0]) if s else None
        except Exception:
            return None

    conn.create_function("LOG", 1, sqlite_log)
    conn.create_function("EXP", 1, sqlite_exp)
    conn.create_function("ASCII", 1, sqlite_ascii)

    # Register aggregate functions
    conn.create_aggregate("VARIANCE", 1, VarianceAggregate)
    conn.create_aggregate("VAR_SAMP", 1, VarianceAggregate)
    conn.create_aggregate("VAR_POP", 1, VariancePopAggregate)
    conn.create_aggregate("STDDEV", 1, StddevAggregate)
    conn.create_aggregate("STDDEV_SAMP", 1, StddevAggregate)
    conn.create_aggregate("STDDEV_POP", 1, StddevPopAggregate)

def get_db_connection() -> sqlite3.Connection:
    """Returns a connection to the persistent SQLite database with registered custom functions."""
    if not DB_PATH.exists() or DB_PATH.stat().st_size == 0:
        create_and_seed_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    register_custom_functions(conn)
    return conn

