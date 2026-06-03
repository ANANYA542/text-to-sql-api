from __future__ import annotations

import os
import time
import logging
import requests
import re
import json
from pathlib import Path
from dotenv import load_dotenv
from app.database.connection import get_db_connection

logger = logging.getLogger(__name__)

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.1-8b-instant"

# File-based cache for generated SQL queries to avoid rate limits
CACHE_FILE = Path(__file__).resolve().parents[2] / ".cache" / "generated_sql_cache.json"
sql_cache = {}

def load_sql_cache():
    global sql_cache
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                sql_cache = json.load(f)
            logger.info(f"Loaded {len(sql_cache)} queries from SQL cache.")
        except Exception as e:
            logger.warning(f"Failed to load SQL cache: {e}")
            sql_cache = {}

def save_sql_cache():
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(sql_cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save SQL cache: {e}")

load_sql_cache()

FEW_SHOT_EXAMPLES = """
Example 1:
Question: For each department, what is the maximum number of enrolled students in any course offered by that department in 2019, what is the name of the responsible faculty member for that course, and what is the average number of enrolled students for all such courses in that department, considering only courses where the maximum number of enrolled students is greater than 5, and only including departments of Mathematics (Course 18) and Chemistry (Course 5)? Do not return any rounded answers.
SQL:
WITH inner_cte AS (
  SELECT sd.DEPARTMENT_NAME, lso.NUM_ENROLLED_STUDENTS AS MAX_NUM_ENROLLED_STUDENTS, lso.RESPONSIBLE_FACULTY_NAME, lso.OFFER_DEPT_CODE
  FROM SIS_DEPARTMENT sd
  JOIN LIBRARY_SUBJECT_OFFERED lso ON sd.DEPARTMENT_CODE = lso.OFFER_DEPT_CODE
  WHERE lso.TERM_CODE LIKE '2019%'
    AND ( (sd.DEPARTMENT_NAME = 'Mathematics') OR (lso.OFFER_DEPT_NAME = 'Chemistry') )
    AND lso.NUM_ENROLLED_STUDENTS > 5
    AND lso.NUM_ENROLLED_STUDENTS = (
      SELECT MAX(lso2.NUM_ENROLLED_STUDENTS)
      FROM LIBRARY_SUBJECT_OFFERED lso2
      WHERE lso2.OFFER_DEPT_CODE = lso.OFFER_DEPT_CODE
        AND lso2.TERM_CODE LIKE '2019%'
        AND lso2.NUM_ENROLLED_STUDENTS > 5
    )
)
SELECT ic.DEPARTMENT_NAME, ic.MAX_NUM_ENROLLED_STUDENTS, ic.RESPONSIBLE_FACULTY_NAME, avg_stats.avg_num_enrolled_students
FROM inner_cte ic
JOIN (
  SELECT lso.OFFER_DEPT_CODE, AVG(lso.NUM_ENROLLED_STUDENTS) AS avg_num_enrolled_students
  FROM LIBRARY_SUBJECT_OFFERED lso
  WHERE lso.TERM_CODE LIKE '2019%'
    AND lso.NUM_ENROLLED_STUDENTS > 5
    AND (lso.OFFER_DEPT_CODE = 18 OR lso.OFFER_DEPT_CODE = 5)
  GROUP BY lso.OFFER_DEPT_CODE
) avg_stats ON ic.OFFER_DEPT_CODE = avg_stats.OFFER_DEPT_CODE
ORDER BY ic.DEPARTMENT_NAME;

Example 2:
Question: For each assignable organization (excluding 'ADMISSIONS' and those with organization IDs '138' and '250'), provide the organization name, the average and variance of the HR department codes for its associated HR org units, and the number of HR org units per organization, considering only those with a non-null HR department code. Do not return any rounded answers.
SQL:
WITH inner_cte AS (
  SELECT FCLT_ORGANIZATION.ORGANIZATION_NAME, HR_ORG_UNIT.HR_ORG_UNIT_ID, HR_ORG_UNIT.HR_DEPARTMENT_CODE_OLD
  FROM FCLT_ORGANIZATION
  JOIN HR_ORG_UNIT ON FCLT_ORGANIZATION.HR_ORG_UNIT_ID = HR_ORG_UNIT.HR_ORG_UNIT_ID
  WHERE FCLT_ORGANIZATION.ASSIGNABLE = 1
    AND HR_ORG_UNIT.HR_DEPARTMENT_CODE_OLD IS NOT NULL
    AND FCLT_ORGANIZATION.ORGANIZATION_NAME != 'ADMISSIONS'
    AND FCLT_ORGANIZATION.ORGANIZATION_ID NOT IN ('138', '250')
)
SELECT inner_cte.ORGANIZATION_NAME, AVG(inner_cte.HR_DEPARTMENT_CODE_OLD) AS avg_dept_code,
       VARIANCE(inner_cte.HR_DEPARTMENT_CODE_OLD) AS dept_code_variance, COUNT(inner_cte.HR_ORG_UNIT_ID) AS num_hr_org_units
FROM inner_cte
GROUP BY inner_cte.ORGANIZATION_NAME
ORDER BY dept_code_variance DESC;

Example 3:
Question: For each school, what is the standard deviation (using STDDEV only and never STDDEV_POP) of the department budget codes among its departments, what is the full name of the school, and how many unique subject codes are associated with the departments in that school? Only include schools that have a department named "Political Science". Do not return any rounded answers.
SQL:
WITH inner_cte AS (
  SELECT sd.SCHOOL_NAME, STDDEV(sd.DEPT_BUDGET_CODE) AS dept_budget_code_stddev, sd.SCHOOL_NAME AS school_full_name
  FROM SIS_DEPARTMENT sd
  WHERE sd.SCHOOL_NAME IN (
    SELECT SCHOOL_NAME FROM SIS_DEPARTMENT WHERE DEPARTMENT_NAME = 'Political Science'
  )
  GROUP BY sd.SCHOOL_NAME
  HAVING COUNT(sd.DEPT_BUDGET_CODE) > 0
)
SELECT icte.SCHOOL_NAME, icte.dept_budget_code_stddev, icte.school_full_name, COUNT(DISTINCT ssc.SUBJECT_CODE) AS unique_subject_codes
FROM inner_cte icte
LEFT JOIN SIS_DEPARTMENT sd ON icte.SCHOOL_NAME = sd.SCHOOL_NAME
LEFT JOIN SIS_SUBJECT_CODE ssc ON sd.DEPARTMENT_CODE = ssc.DEPARTMENT_CODE
GROUP BY icte.SCHOOL_NAME, icte.dept_budget_code_stddev, icte.school_full_name;
"""

def get_table_schema_ddl(table_name: str) -> str:
    """Gets the column names for a table in a highly compact format."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"PRAGMA table_info(\"{table_name}\")")
        columns = cursor.fetchall()
        col_parts = []
        for col in columns:
            col_name = col["name"]
            # Ignore audit/metadata columns to save tokens
            c = col_name.lower()
            if any(x in c for x in ("warehouse_load_date", "last_activity_date", "load_date", "record_status", "load_dt", "update_date")):
                continue
            col_parts.append(col_name)
        return f"{table_name}({', '.join(col_parts)})"
    except Exception as e:
        logger.error(f"Error fetching schema info for table {table_name}: {e}")
        return f"{table_name}()"
    finally:
        conn.close()

last_api_call_time = 0.0

def throttle_api_call(min_interval=7.0):
    global last_api_call_time
    now = time.time()
    elapsed = now - last_api_call_time
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        logger.info(f"Throttling API call in SQL generation. Sleeping for {sleep_time:.2f}s...")
        time.sleep(sleep_time)
    last_api_call_time = time.time()

def generate_sql_query(question: str, retrieved_tables: list[str]) -> str:
    """
    Generates SQLite dialect SQL from natural language question and retrieved table list
    using Groq API model llama-3.1-8b-instant.
    """
    if question in sql_cache:
        logger.info(f"Returning cached SQL for question: '{question}'")
        return sql_cache[question]

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")

    schema_lines = []
    for table in retrieved_tables:
        schema_lines.append(get_table_schema_ddl(table))
    schema_context = "\n".join(schema_lines)

    # Cleaned and optimized system instruction
    system_instruction = (
        "You are a SQLite expert. Return ONLY the raw SQL query.\n"
        "No markdown. No explanation. No code blocks. Just SQL."
    )

    user_prompt = (
        f"SCHEMA:\n{schema_context}\n\n"
        f"EXAMPLES:\n{FEW_SHOT_EXAMPLES.strip()}\n\n"
        f"QUESTION: {question}\n"
        f"SQL:"
    )

    retries = 3
    backoff_delays = [2, 4, 8]
    
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
            if attempt == 0:
                throttle_api_call(min_interval=7.0)
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=15
            )
            if response.status_code == 200:
                result_json = response.json()
                sql_text = result_json["choices"][0]["message"]["content"].strip()
            
                # Aggressively clean up markdown code block wrapper
                if "```" in sql_text:
                    parts = sql_text.split("```")
                    if len(parts) >= 3:
                        code_part = parts[1]
                        if code_part.lower().startswith("sql"):
                            code_part = code_part[3:]
                        sql_text = code_part.strip()
                    else:
                        lines = sql_text.split("\n")
                        lines = [l for l in lines if not l.strip().startswith("```")]
                        sql_text = "\n".join(lines).strip()
                
                # Double-check that it does not contain system prefixes
                if sql_text.upper().startswith("SQL:"):
                    sql_text = sql_text[4:].strip()
                sql_cache[question] = sql_text
                save_sql_cache()
                return sql_text
            elif response.status_code == 429:
                retry_after = backoff_delays[attempt] if attempt < len(backoff_delays) else 10
                try:
                    err_detail = response.json()
                    msg = err_detail.get("error", {}).get("message", "")
                    if "try again in" in msg:
                        match = re.search(r"try again in ([\d\.]+)s", msg)
                        if match:
                            retry_after = float(match.group(1)) + 0.5
                except Exception:
                    pass
                logger.warning(
                    f"Groq API rate limit (429) hit. "
                    f"Retrying in {retry_after}s... Attempt {attempt + 1}/{retries + 1}"
                )
                time.sleep(retry_after)
            else:
                logger.warning(
                    f"Groq API returned error status {response.status_code}: {response.text}. "
                    f"Attempt {attempt + 1}/{retries + 1}"
                )
                if attempt < retries:
                    time.sleep(backoff_delays[attempt])
        except Exception as e:
            logger.warning(f"Error calling Groq API: {e}. Attempt {attempt + 1}/{retries + 1}")
            if attempt < retries:
                time.sleep(backoff_delays[attempt])

    raise RuntimeError("Failed to generate SQL via Groq API after multiple retries.")
