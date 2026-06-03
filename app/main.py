from __future__ import annotations

import os
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

from app.core import config
from app.models.requests import RetrieveRequest, GenerateSQLRequest
from app.models.responses import (
    RetrieveResponse, TableDetail, GenerateSQLResponse,
    BenchmarkResponse, BenchmarkMetrics, BenchmarkSubtaskBreakdown, BenchmarkErrorAnalysis,
    ErrorResponse
)
from app.retrieval.schema_loader import load_beaver_schema, load_beaver_queries
from app.retrieval.engine import RetrievalEngine
from app.generation.generator import generate_sql_query
from app.database.validator import validate_sql
from app.database.connection import get_db_connection, create_and_seed_db
from scripts.test_retrieval_accuracy import extract_tables_and_ctes

logger = logging.getLogger(__name__)

# Global state
engine: RetrievalEngine | None = None
valid_schema_tables: set[str] = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, valid_schema_tables
    logger.info("=== Starting up: initializing database and models ===")
    try:
        # Create and seed SQLite database
        create_and_seed_db()
        
        # Load schema
        schema = load_beaver_schema(split="dw")
        valid_schema_tables = {name.upper() for name in schema.keys()}
        
        # Initialize retrieval engine
        engine = RetrievalEngine(schema)
        logger.info("=== Startup complete ===")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    yield
    logger.info("=== Shutting down ===")

app = FastAPI(
    title="Enterprise Text-to-SQL API",
    description="Full-featured enterprise Text-to-SQL system using Beaver benchmark.",
    version="1.0.0",
    lifespan=lifespan,
)

@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def retrieve_tables(request: RetrieveRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Retrieval engine not initialized")
    try:
        result = engine.retrieve(request.question, top_k=request.top_k)
        return RetrieveResponse(**result)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/generate-sql",
    response_model=GenerateSQLResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def generate_sql(request: GenerateSQLRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Retrieval engine not initialized")
    
    start_time = time.time()
    try:
        # 1. Retrieve tables
        ret_val = engine.retrieve(request.question, top_k=5)
        retrieved_tables = ret_val.get("retrieved_tables", [])
        confidence = ret_val.get("confidence", 0.5)

        # 2. Generate SQL query
        sql_query = generate_sql_query(request.question, retrieved_tables)

        # 3. Validate generated SQL
        is_valid_syntax, parsing_errors = validate_sql(sql_query)

        # Build prompt used for debugging/audit
        prompt_used = f"Question: {request.question} | Context tables: {', '.join(retrieved_tables)}"

        return GenerateSQLResponse(
            sql=sql_query,
            retrieved_tables=retrieved_tables,
            is_valid_syntax=is_valid_syntax,
            parsing_errors=parsing_errors,
            confidence=confidence,
            prompt_used=prompt_used
        )
    except Exception as e:
        logger.error(f"SQL generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post(
    "/benchmark",
    response_model=BenchmarkResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def run_benchmark():
    if engine is None:
        raise HTTPException(status_code=500, detail="Retrieval engine not initialized")

    logger.info("Starting benchmark evaluation...")
    
    try:
        query_ds = load_beaver_queries(split="dw")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load benchmark dataset: {e}")

    # Load 25 queries for benchmark evaluation
    eval_queries = []
    for row in query_ds:
        if len(eval_queries) >= 25:
            break
        if row.get("question") and row.get("sql"):
            eval_queries.append(row)

    total_evaluated = 0
    total_recall_5 = 0.0
    total_recall_10 = 0.0
    exact_matches = 0
    execution_matches = 0
    parsing_successes = 0
    total_latency_ms = 0.0

    # Subtask tracking
    subtasks = {
        "multi_table": {"count": 0, "correct": 0},
        "column_mapping": {"count": 0, "correct": 0},
        "join_detection": {"count": 0, "correct": 0},
        "domain_knowledge": {"count": 0, "correct": 0}
    }

    # Error analysis counters
    retrieval_failures = 0
    parsing_failures = 0
    execution_failures = 0
    logic_errors = 0

    conn = get_db_connection()

    for idx, row in enumerate(eval_queries):
        question = row["question"]
        gold_sql = row["sql"]

        # Parse tables and CTEs from gold SQL
        raw_parsed_tables, ctes = extract_tables_and_ctes(gold_sql)

        # Filter out CTE names and verify against physical schema
        tables_no_ctes = raw_parsed_tables - ctes
        gold_tables = {t for t in tables_no_ctes if t in valid_schema_tables}

        # If zero real gold tables after filtering, skip entirely
        if not gold_tables:
            logger.info(f"Skipping query #{idx} with zero real gold tables after CTE filter.")
            continue

        total_evaluated += 1
        start_time = time.perf_counter()

        # 1. Retrieve
        ret_val_5 = engine.retrieve(question, top_k=5)
        ret_tables_5 = [t.upper() for t in ret_val_5.get("retrieved_tables", [])]
        ret_val_10 = engine.retrieve(question, top_k=10)
        ret_tables_10 = [t.upper() for t in ret_val_10.get("retrieved_tables", [])]

        # Compute Recalls
        matched_5 = gold_tables.intersection(set(ret_tables_5))
        recall_5 = len(matched_5) / len(gold_tables)
        total_recall_5 += recall_5

        matched_10 = gold_tables.intersection(set(ret_tables_10))
        recall_10 = len(matched_10) / len(gold_tables)
        total_recall_10 += recall_10

        if recall_5 == 0.0:
            retrieval_failures += 1

        # Determine subtask categories
        is_multi_table = len(gold_tables) > 1
        has_join = "JOIN" in gold_sql.upper()
        has_column = any(x in question.lower() for x in ["budget", "phone", "email", "name", "title"])
        has_domain = any(x in question.lower() for x in ["mit", "residential", "chemical", "biology", "mathematics"])

        if is_multi_table:
            subtasks["multi_table"]["count"] += 1
        if has_join:
            subtasks["join_detection"]["count"] += 1
        if has_column:
            subtasks["column_mapping"]["count"] += 1
        if has_domain:
            subtasks["domain_knowledge"]["count"] += 1

        # 2. Generate SQL
        generated_sql = ""
        is_parsed = False
        is_exact = False
        is_exec = False

        try:
            generated_sql = generate_sql_query(question, ret_val_5.get("retrieved_tables", []))
            
            # Syntax Parse check
            is_valid_syntax, _ = validate_sql(generated_sql)
            if is_valid_syntax:
                is_parsed = True
                parsing_successes += 1
            else:
                parsing_failures += 1

            # Exact Match check (whitespace/case normalized)
            norm_gen = "".join(generated_sql.lower().split())
            norm_gold = "".join(gold_sql.lower().split())
            if norm_gen == norm_gold:
                is_exact = True
                exact_matches += 1

            # Execution Match check
            cursor = conn.cursor()
            gold_rows = []
            gen_rows = []

            # Run gold SQL
            try:
                cursor.execute(gold_sql)
                gold_rows = [tuple(r) for r in cursor.fetchall()]
            except Exception as e:
                # Gold SQL execution error (should be rare)
                logger.warning(f"Failed to execute gold SQL on beaver_dw.db: {e}")
                gold_rows = []

            # Run generated SQL
            try:
                cursor.execute(generated_sql)
                gen_rows = [tuple(r) for r in cursor.fetchall()]
                
                # Check match
                if set(gen_rows) == set(gold_rows):
                    is_exec = True
                    execution_matches += 1
                else:
                    logic_errors += 1
            except Exception as e:
                execution_failures += 1
                logger.warning(f"Generated SQL execution failed: {e}")

        except Exception as e:
            logger.error(f"Error evaluating query #{idx}: {e}")
            parsing_failures += 1

        # Update subtask performance
        if is_exec:
            if is_multi_table:
                subtasks["multi_table"]["correct"] += 1
            if has_join:
                subtasks["join_detection"]["correct"] += 1
            if has_column:
                subtasks["column_mapping"]["correct"] += 1
            if has_domain:
                subtasks["domain_knowledge"]["correct"] += 1

        latency = (time.perf_counter() - start_time) * 1000.0
        total_latency_ms += latency

    conn.close()

    if total_evaluated == 0:
        raise HTTPException(status_code=500, detail="No queries were evaluated during benchmark.")

    # Calculate final averages
    metrics = BenchmarkMetrics(
        retrieval_recall_at_5=round(total_recall_5 / total_evaluated, 4),
        retrieval_recall_at_10=round(total_recall_10 / total_evaluated, 4),
        sql_exact_match_accuracy=round(exact_matches / total_evaluated, 4),
        sql_execution_match_accuracy=round(execution_matches / total_evaluated, 4),
        parsing_success_rate=round(parsing_successes / total_evaluated, 4),
        average_latency_ms=round(total_latency_ms / total_evaluated, 2)
    )

    # Subtask breakdown helpers
    def get_subtask_pct(sub_key: str) -> float:
        total = subtasks[sub_key]["count"]
        return round(subtasks[sub_key]["correct"] / total, 4) if total > 0 else 1.0

    breakdown = BenchmarkSubtaskBreakdown(
        multi_table_retrieval=get_subtask_pct("multi_table"),
        column_mapping=get_subtask_pct("column_mapping"),
        join_detection=get_subtask_pct("join_detection"),
        domain_knowledge=get_subtask_pct("domain_knowledge")
    )

    errors = BenchmarkErrorAnalysis(
        retrieval_failures=retrieval_failures,
        parsing_failures=parsing_failures,
        execution_failures=execution_failures,
        logic_errors=logic_errors
    )

    return BenchmarkResponse(
        total_queries=total_evaluated,
        metrics=metrics,
        subtask_breakdown=breakdown,
        error_analysis=errors
    )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "num_tables": len(engine.table_names) if engine else 0,
        "database_exists": os.path.exists(config.DB_PATH)
    }
