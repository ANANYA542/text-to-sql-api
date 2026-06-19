from __future__ import annotations

import os
import time
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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
from app.core.metrics_collector import MetricsCollector, RequestRecord
from app.core.pipeline_logger import PipelineLogger
from scripts.test_retrieval_accuracy import extract_tables_and_ctes

logger = logging.getLogger(__name__)

# Global state
engine: RetrievalEngine | None = None
valid_schema_tables: set[str] = set()
schema_data: dict[str, str] = {}
metrics_collector = MetricsCollector()
pipeline_logger = PipelineLogger()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, valid_schema_tables, schema_data
    logger.info("=== Starting up: initializing database and models ===")
    try:
        # Create and seed SQLite database
        create_and_seed_db()
        
        # Load schema
        schema_data = load_beaver_schema(split="dw")
        valid_schema_tables = {name.upper() for name in schema_data.keys()}
        
        # Initialize retrieval engine
        engine = RetrievalEngine(schema_data)
        logger.info("=== Startup complete ===")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    yield
    logger.info("=== Shutting down ===")

app = FastAPI(
    title="Enterprise Text-to-SQL API",
    description="Full-featured enterprise Text-to-SQL system using Beaver benchmark.",
    version="2.0.0",
    lifespan=lifespan,
)

# Mount static files for the dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ═══════════════════════════════════════════════════════════════
# Dashboard Route
# ═══════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def serve_dashboard():
    """Serves the interactive web dashboard."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Dashboard not found. Visit /docs for API documentation."}

# ═══════════════════════════════════════════════════════════════
# Core API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.post(
    "/retrieve",
    response_model=RetrieveResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def retrieve_tables(request: RetrieveRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Retrieval engine not initialized")
    
    start_time = time.perf_counter()
    try:
        result = engine.retrieve(request.question, top_k=request.top_k)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Record metrics
        metrics_collector.record(RequestRecord(
            endpoint="/retrieve",
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=True,
            num_tables_retrieved=len(result.get("retrieved_tables", [])),
            top_confidence=result.get("confidence", 0.0),
            used_learned_ranker="learned" in result.get("model_used", ""),
            retrieval_ms=result.get("latency_breakdown", {}).get("retrieval_ms", 0),
            reranking_ms=result.get("latency_breakdown", {}).get("reranking_ms", 0),
        ))
        
        # Log pipeline execution
        pipeline_logger.log(
            question=request.question,
            retrieved_tables=result.get("retrieved_tables", []),
            scores=result.get("scores", []),
            confidence=result.get("confidence", 0.0),
            model_used=result.get("model_used", "heuristic"),
            latency_ms=latency_ms,
            latency_breakdown=result.get("latency_breakdown", {}),
        )
        
        return RetrieveResponse(**result)
    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000
        metrics_collector.record(RequestRecord(
            endpoint="/retrieve",
            timestamp=time.time(),
            latency_ms=latency_ms,
            success=False,
            error_type=type(e).__name__,
        ))
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
    
    start_time = time.perf_counter()
    try:
        # 1. Retrieve tables
        t_retrieve = time.perf_counter()
        ret_val = engine.retrieve(request.question, top_k=5)
        retrieved_tables = ret_val.get("retrieved_tables", [])
        confidence = ret_val.get("confidence", 0.5)
        retrieval_ms = (time.perf_counter() - t_retrieve) * 1000

        # 2. Generate SQL query
        t_gen = time.perf_counter()
        sql_query = generate_sql_query(request.question, retrieved_tables)
        generation_ms = (time.perf_counter() - t_gen) * 1000

        # 3. Validate generated SQL
        t_val = time.perf_counter()
        is_valid_syntax, parsing_errors = validate_sql(sql_query)
        validation_ms = (time.perf_counter() - t_val) * 1000

        # Build prompt used for debugging/audit
        prompt_used = f"Question: {request.question} | Context tables: {', '.join(retrieved_tables)}"

        total_latency_ms = (time.perf_counter() - start_time) * 1000

        # Record metrics
        metrics_collector.record(RequestRecord(
            endpoint="/generate-sql",
            timestamp=time.time(),
            latency_ms=total_latency_ms,
            success=True,
            num_tables_retrieved=len(retrieved_tables),
            top_confidence=confidence,
            used_learned_ranker="learned" in ret_val.get("model_used", ""),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            validation_ms=validation_ms,
        ))

        # Log pipeline execution
        pipeline_logger.log(
            question=request.question,
            retrieved_tables=retrieved_tables,
            scores=ret_val.get("scores", []),
            confidence=confidence,
            model_used=ret_val.get("model_used", "heuristic"),
            sql_generated=sql_query,
            is_valid=is_valid_syntax,
            parsing_errors=parsing_errors,
            latency_ms=total_latency_ms,
            latency_breakdown={
                "retrieval_ms": round(retrieval_ms, 2),
                "generation_ms": round(generation_ms, 2),
                "validation_ms": round(validation_ms, 2),
            },
        )

        return GenerateSQLResponse(
            sql=sql_query,
            retrieved_tables=retrieved_tables,
            is_valid_syntax=is_valid_syntax,
            parsing_errors=parsing_errors,
            confidence=confidence,
            prompt_used=prompt_used
        )
    except Exception as e:
        total_latency_ms = (time.perf_counter() - start_time) * 1000
        metrics_collector.record(RequestRecord(
            endpoint="/generate-sql",
            timestamp=time.time(),
            latency_ms=total_latency_ms,
            success=False,
            error_type=type(e).__name__,
        ))
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

# ═══════════════════════════════════════════════════════════════
# Dashboard API Endpoints
# ═══════════════════════════════════════════════════════════════

@app.get("/api/schema")
async def get_schema():
    """Returns all table schemas for the schema explorer."""
    if not schema_data:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()
    
    tables = []
    for table_name, desc in schema_data.items():
        # Get column names from the database
        columns = []
        try:
            cursor.execute(f'PRAGMA table_info("{table_name}")')
            for col in cursor.fetchall():
                col_name = col["name"]
                # Skip audit columns
                c = col_name.lower()
                if any(x in c for x in ("warehouse_load_date", "last_activity_date", "load_date", "load_dt")):
                    continue
                columns.append(col_name)
        except Exception:
            pass

        # Count relations
        from app.retrieval.engine import RetrievalEngine
        relations_count = 0
        if engine:
            neighbors = engine.relations.get(table_name.upper(), [])
            relations_count = len(neighbors)

        tables.append({
            "table_name": table_name,
            "columns": columns,
            "relations": relations_count,
            "description": desc[:200] if desc else "",
        })

    conn.close()
    return tables

@app.get("/api/experiments")
async def get_experiments():
    """Returns ML experiment history for the dashboard."""
    from app.ml.experiment_tracker import ExperimentTracker
    
    tracker = ExperimentTracker()
    runs = tracker.list_runs()
    
    return [
        {
            "run_id": run.run_id,
            "model_type": run.model_type,
            "timestamp": run.timestamp,
            "num_samples": run.num_training_samples,
            "training_duration_s": round(run.training_duration_seconds, 1),
            "feature_importances": run.feature_importances,
            **run.val_metrics,
        }
        for run in runs
    ]

@app.get("/api/metrics")
async def get_metrics():
    """Returns production metrics for the monitoring dashboard."""
    return metrics_collector.get_metrics()

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "num_tables": len(engine.table_names) if engine else 0,
        "database_exists": os.path.exists(config.DB_PATH)
    }
