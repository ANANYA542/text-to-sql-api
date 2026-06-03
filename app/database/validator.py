from __future__ import annotations

import logging
import sqlglot
from app.database.connection import get_db_connection

logger = logging.getLogger(__name__)

def validate_sql(sql: str) -> tuple[bool, str | None]:
    """
    Validates a SQL query.
    1. Parses it using sqlglot to ensure basic syntax is valid.
    2. Runs an EXPLAIN QUERY PLAN on the database to check if tables,
       columns, and Joins compile correctly.
    
    Returns:
        (is_valid: bool, error_message: str | None)
    """

    try:
        sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:
        error_msg = f"SQL parsing error: {e}"
        logger.warning(error_msg)
        return False, error_msg

    # 2. Database validation using EXPLAIN QUERY PLAN
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
       
        cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
        cursor.fetchall()
        return True, None
    except Exception as e:
        error_msg = f"SQL execution/compilation error: {e}"
        logger.warning(error_msg)
        return False, error_msg
    finally:
        conn.close()
