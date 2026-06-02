#!/usr/bin/env python3
"""
Standalone evaluation script to measure Text-to-SQL Retrieval Engine Recall@5 accuracy.

Workflow:
1. Load 50 queries from the Hugging Face beaver-query dataset (dw split).
2. For each query, parse the ground truth SQL using sqlglot:
   - Extract all defined Common Table Expression (CTE) names from the WITH clause.
   - Extract all referenced table names, and exclude CTEs.
   - Query the local active FastAPI Retrieval API (/health) to get the list of 97 valid database tables.
   - Filter the parsed gold tables to keep only tables that actually exist in the database schema.
3. Query the local active FastAPI Retrieval API (/retrieve) to retrieve the top 5 tables.
4. Calculate Recall@5 per question:
   Recall@5 = (number of real gold tables present in top 5) / (total real gold tables)
5. Average Recall@5 across all processed queries (excluding queries with zero real gold tables).
6. Display detailed breakdown, final percentage score, and the 10 worst performing queries.
"""

import os
from pathlib import Path
import json
import logging
import urllib.request
import urllib.error

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datasets import load_dataset
import sqlglot
from sqlglot import exp

# Set up clean logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
BASE_URL = os.getenv("RETRIEVAL_API_URL", "http://localhost:8000")

def extract_tables_and_ctes(sql_query: str) -> tuple[set[str], set[str]]:
    """
    Parses SQL using sqlglot and extracts:
    - Unique raw table names referenced in the query.
    - CTE names defined in WITH clauses.
    """
    try:
        parsed = sqlglot.parse_one(sql_query)
        tables = set()
        ctes = set()

        # Extract CTE names from WITH clauses
        for cte in parsed.find_all(exp.CTE):
            cte_name = cte.alias_or_name.upper()
            if cte_name:
                ctes.add(cte_name)

        # Extract all table names in AST
        for node in parsed.find_all(exp.Table):
            t_name = node.name.upper()
            if t_name:
                tables.add(t_name)

        return tables, ctes
    except Exception as e:
        logger.warning(f"Failed to parse SQL via sqlglot: {e}. Falling back to basic parsing.")
        # Fallback simple parsing
        tables = set()
        ctes = set()
        words = sql_query.upper().replace('`', '').replace('"', '').split()
        for i, word in enumerate(words):
            if word == "WITH":
                # Very simple heuristic for CTE name following WITH
                if i + 1 < len(words):
                    ctes.add(words[i+1].strip())
            elif word in ("FROM", "JOIN") and i + 1 < len(words):
                next_word = words[i+1].strip().split('(')[0].split(')')[0]
                if next_word and not next_word.startswith("SELECT") and next_word not in ("(", "SELECT"):
                    tables.add(next_word.split('.')[-1])
        return tables, ctes

def get_valid_schema_tables() -> set[str]:
    """
    Retrieves the list of valid schema table names loaded in the running FastAPI server.
    """
    url = f"{BASE_URL}/health"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode())
            # If the endpoint doesn't explicitly return all table names, we can query retrieve once
            # to trigger load if needed, or fallback.
            # But wait, let's call /retrieve with a dummy query to fetch a sample to get table names
            # from retrieval engine in python context if health doesn't give them.
            # However, we can construct the list of 97 tables from the RetrievalEngine inside retrieve.
            return None
    except Exception as e:
        logger.error(f"Failed to query server health: {e}")
        return None

def get_tables_from_api_retrieve() -> set[str]:
    """
    Retrieves the exact list of 97 schema tables by hitting a special check or fallback.
    Since we know the 97 tables are physical database tables, let's load them via the loader
    or fetch them from a test query retrieval.
    """
    # Simply load beaver-table directly using datasets since we have HF logged in!
    try:
        table_ds = load_dataset("beaverbench/beaver-table", split="dw")
        return {row["table_name"].upper() for row in table_ds}
    except Exception as e:
        logger.error(f"Could not load schema tables list from HF: {e}")
        # Return hardcoded typical Beaver DW tables list if offline fallback needed
        return set()

def query_retrieval_api(question: str, top_k: int = 5) -> list[str]:
    """
    Calls the locally running FastAPI retrieval endpoint to fetch the top_k tables.
    """
    url = f"{BASE_URL}/retrieve"
    data = json.dumps({"question": question, "top_k": top_k}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            res = json.loads(response.read().decode())
            return res.get("retrieved_tables", [])
    except urllib.error.URLError as e:
        logger.error(f"Failed to connect to local API server at {url}: {e}")
        logger.info(
            f"Please make sure the FastAPI server is running (e.g. uvicorn main:app --port 8000, or set RETRIEVAL_API_URL={BASE_URL})"
        )
        raise RuntimeError("FastAPI server connection failed") from e

def run_evaluation():
    logger.info("Starting Text-to-SQL Retrieval Accuracy Evaluation...")

    # Fetch valid schema tables
    valid_schema_tables = get_tables_from_api_retrieve()
    logger.info(f"Loaded {len(valid_schema_tables)} valid physical schema tables.")

    # 1. Load the Beaver query dataset (dw split) from Hugging Face
    logger.info("Loading beaver-query dataset split=dw...")
    try:
        query_ds = load_dataset("beaverbench/beaver-query", split="dw")
    except Exception as e:
        logger.error(f"Failed to load Hugging Face dataset: {e}")
        return

    # Slice first 50 queries having ground truth SQL answers
    eval_queries = []
    for row in query_ds:
        if len(eval_queries) >= 50:
            break
        if row.get("question") and row.get("sql"):
            eval_queries.append(row)

    logger.info(f"Successfully loaded {len(eval_queries)} queries for evaluation.")

    # 2. Evaluate each question
    results = []
    total_recall = 0.0
    zero_gold_table_queries_count = 0

    for idx, row in enumerate(eval_queries):
        question = row["question"]
        gold_sql = row["sql"]

        # Parse tables and CTEs from gold SQL
        raw_parsed_tables, ctes = extract_tables_and_ctes(gold_sql)

        # 1. Filter out CTEs
        gold_before_filter = list(raw_parsed_tables)
        ctes_removed = raw_parsed_tables.intersection(ctes)
        tables_no_ctes = raw_parsed_tables - ctes

        # 2. Keep only tables that actually exist in the physical schema
        gold_tables = {t for t in tables_no_ctes if t in valid_schema_tables}
        non_existent_tables = tables_no_ctes - gold_tables

        if not gold_tables:
            logger.info(f"Skipping Q#{idx} - has ZERO physical gold tables after CTE and schema filtering.")
            zero_gold_table_queries_count += 1
            continue

        # Get retrieved tables from API
        try:
            retrieved_tables = query_retrieval_api(question, top_k=5)
        except Exception:
            logger.error("Stopping evaluation due to API connection failures.")
            return

        # Normalize retrieved table names to uppercase
        retrieved_upper = [t.upper() for t in retrieved_tables]

        # Calculate recall@5
        matched_tables = gold_tables.intersection(set(retrieved_upper))
        recall_score = len(matched_tables) / len(gold_tables) if gold_tables else 0.0
        total_recall += recall_score

        results.append({
            "index": idx,
            "question": question,
            "gold_before_filter": gold_before_filter,
            "ctes_removed": list(ctes_removed),
            "non_existent_removed": list(non_existent_tables),
            "gold_tables": list(gold_tables),
            "retrieved_tables": retrieved_upper,
            "recall": recall_score,
            "matched": list(matched_tables)
        })

    if not results:
        logger.error("No queries were successfully evaluated.")
        return

    # 3. Print average Recall@5 score
    avg_recall_percentage = (total_recall / len(results)) * 100
    print("\n" + "="*80)
    print(f"EVALUATION COMPLETE: FINAL Recall@5 = {avg_recall_percentage:.2f}%")
    print(f"Total Evaluated Questions: {len(results)}")
    print(f"Questions Skipped (Zero Real Gold Tables after CTE/Schema filter): {zero_gold_table_queries_count}")
    print("="*80 + "\n")

    # 4. Print Per-Question Breakdown
    print("DETAILED BREAKDOWN:")
    for res in results:
        print(f"Q#{res['index']}: {res['question']}")
        print(f"  Gold Tables (BEFORE Filter): {res['gold_before_filter']}")
        print(f"  CTEs Removed:                {res['ctes_removed']}")
        print(f"  Non-existent Schema Removed: {res['non_existent_removed']}")
        print(f"  Gold Tables (AFTER Filter):  {res['gold_tables']}")
        print(f"  Retrieved Tables:            {res['retrieved_tables']}")
        print(f"  Recall@5:                    {res['recall'] * 100:.1f}% (Matched {len(res['matched'])}/{len(res['gold_tables'])})")
        print("-" * 50)

    # 5. Print Top 10 worst performing questions (sorted by lowest recall)
    worst_questions = sorted(results, key=lambda x: x["recall"])[:10]
    print("\n" + "="*80)
    print("TOP 10 WORST PERFORMING QUESTIONS:")
    print("="*80)
    for w in worst_questions:
        print(f"Q#{w['index']} (Recall@5 = {w['recall']*100:.1f}%):")
        print(f"  Question:     {w['question']}")
        print(f"  Gold Tables:  {w['gold_tables']}")
        print(f"  Retrieved:    {w['retrieved_tables']}")
        print("-" * 50)

if __name__ == "__main__":
    run_evaluation()
