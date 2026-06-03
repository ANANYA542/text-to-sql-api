from __future__ import annotations

import os
import time
import logging
import requests
from dotenv import load_dotenv
from app.database.connection import get_db_connection

logger = logging.getLogger(__name__)


load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

FEW_SHOT_EXAMPLES = """
Example 1:
Question: For each department, list the department name and the total number of subjects offered by that department in 2022.
SQL:
SELECT t1.DEPARTMENT_NAME, COUNT(t2.SUBJECT_ID) AS TOTAL_SUBJECTS
FROM SIS_DEPARTMENT t1
JOIN SUBJECT_OFFERED_SUMMARY t2 ON t1.DEPARTMENT_CODE = t2.DEPARTMENT_CODE
WHERE t2.ACADEMIC_YEAR = 2022
GROUP BY t1.DEPARTMENT_NAME;

Example 2:
Question: What is the maximum number of rooms for residential buildings located at the MIT site?
SQL:
SELECT MAX(ROOM_COUNT) AS MAX_ROOMS
FROM FCLT_BUILDING
WHERE IS_RESIDENTIAL = 0 AND SITE_NAME = 'MIT Site';

Example 3:
Question: Show the full name of degree-granting departments that have offered more than 5 chemistry subjects.
SQL:
SELECT t1.DEPARTMENT_FULL_NAME
FROM SIS_DEPARTMENT t1
JOIN TIP_SUBJECT_OFFERED t2 ON t1.DEPARTMENT_CODE = t2.DEPARTMENT_CODE
WHERE t1.IS_DEGREE_GRANTING = 1 AND t2.SUBJECT_TITLE LIKE '%Chemistry%'
GROUP BY t1.DEPARTMENT_FULL_NAME
HAVING COUNT(t2.SUBJECT_ID) > 5;
"""

def get_table_schema_ddl(table_name: str) -> str:
    """Gets the column names and inferred SQLite types for a table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
        columns = cursor.fetchall()
        col_parts = []
        for col in columns:
            col_name = col["name"]
            col_type = col["type"]
            col_parts.append(f"{col_name} ({col_type})")
        return f"Table {table_name}: columns = [{', '.join(col_parts)}]"
    except Exception as e:
        logger.error(f"Error fetching schema info for table {table_name}: {e}")
        return f"Table {table_name}: columns = []"
    finally:
        conn.close()

def generate_sql_query(question: str, retrieved_tables: list[str]) -> str:
    """
    Generates SQLite dialect SQL from natural language question and retrieved table list
    using Groq API model llama-3.1-8b-instant.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")


    schema_lines = []
    for table in retrieved_tables:
        schema_lines.append(get_table_schema_ddl(table))
    schema_context = "\n".join(schema_lines)

    # 2. Build prompts conforming to Change 5
    system_instruction = (
        "You are a translation assistant converting natural language questions into raw SQL.\n"
        "Return ONLY the raw SQL query. Do not include markdown code blocks, do not explain the code, "
        "and do not output any text other than the SQL query itself.\n"
        "Always use SQLite syntax only. Do not use PostgreSQL-specific or other non-SQLite database functions.\n"
        "Pay special attention to JOIN keys, aggregation operators, and correct table names."
    )

    user_prompt = (
        f"Available Schema:\n{schema_context}\n\n"
        f"Few-shot Examples:\n{FEW_SHOT_EXAMPLES}\n\n"
        f"User Question: {question}\n"
        f"SQL:"
    )


    retries = 3
    backoff_delays = [1, 2, 4]
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 500
    }

    for attempt in range(retries + 1):
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            if response.status_code == 200:
                result_json = response.json()
                sql_text = result_json["choices"][0]["message"]["content"].strip()
            
                if sql_text.startswith("```"):
                    lines = sql_text.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines[-1].startswith("```"):
                        lines = lines[:-1]
                    sql_text = "\n".join(lines).strip()
                return sql_text
            else:
                logger.warning(
                    f"Groq API returned error status {response.status_code}: {response.text}. "
                    f"Attempt {attempt + 1}/{retries + 1}"
                )
        except Exception as e:
            logger.warning(f"Error calling Groq API: {e}. Attempt {attempt + 1}/{retries + 1}")

        if attempt < retries:
            delay = backoff_delays[attempt]
            logger.info(f"Retrying Groq API call in {delay}s...")
            time.sleep(delay)

    raise RuntimeError("Failed to generate SQL via Groq API after multiple retries.")
