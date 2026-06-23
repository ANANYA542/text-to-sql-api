import streamlit as st
import os
import time
import json
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re

# Configure page settings first
st.set_page_config(
    page_title="Enterprise Text-to-SQL Engine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.core import config
from app.retrieval.schema_loader import load_beaver_schema, load_beaver_queries
from app.retrieval.engine import RetrievalEngine
from app.generation.generator import generate_sql_query
from app.database.validator import validate_sql
from app.database.connection import get_db_connection, create_and_seed_db
from app.core.metrics_collector import MetricsCollector, RequestRecord
from app.core.pipeline_logger import PipelineLogger
from scripts.test_retrieval_accuracy import extract_tables_and_ctes

# Global resource initialization
@st.cache_resource
def get_resources():
    create_and_seed_db()
    schema_data = load_beaver_schema(split="dw")
    engine = RetrievalEngine(schema_data)
    metrics_collector = MetricsCollector()
    pipeline_logger = PipelineLogger()
    return engine, schema_data, metrics_collector, pipeline_logger

engine, schema_data, metrics, pipeline_logger = get_resources()
valid_schema_tables = {name.upper() for name in schema_data.keys()}

# Styling Custom Injections
st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 14px;
        color: #64748B;
        font-weight: 500;
    }
    .custom-title {
        font-size: 32px;
        font-weight: 800;
        color: #0F172A;
        margin-bottom: 4px;
    }
    .custom-subtitle {
        font-size: 16px;
        color: #64748B;
        margin-bottom: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# App header
st.markdown('<div class="custom-title">🤖 Enterprise Text-to-SQL Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-subtitle">Production-grade ML pipeline on Beaver Benchmark with LTR reranking</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🔍 Workspace", 
    "🗂 Schema Explorer", 
    "📊 Evaluation & Benchmark", 
    "🔬 ML Experiments", 
    "📈 System Analytics",
    "📋 Pipeline Logs"
])

# ==========================================
# TAB 1: WORKSPACE
# ==========================================
with tabs[0]:
    st.subheader("Interactive SQL Generation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        question = st.text_area(
            "Natural Language Question",
            value="Which departments have more than 100 students?",
            height=100
        )
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            top_k = st.slider("Top K Tables to Retrieve", min_value=1, max_value=10, value=5)
        with col_ctrl2:
            use_learned = st.toggle("Use Learned ML Ranker (LightGBM/XGBoost)", value=True)
            
        run_btn = st.button("Execute Pipeline", type="primary")
        
    with col2:
        st.info(
            "**How it works**:\n"
            "1. **Query Expansion**: LLM & synonyms expand context.\n"
            "2. **Candidate Retrieval**: Hybrid BM25 + Cosine search (Top 25).\n"
            "3. **Learned Ranker**: Reranks tables using 28 ML features.\n"
            "4. **SQL Gen**: Context-restricted LLM generates optimal query.\n"
            "5. **Execution**: Validated against SQLite in real-time."
        )

    if run_btn and question:
        start_time = time.perf_counter()
        
        # Pipeline Logs tracker
        status_box = st.status("Running pipeline stage-by-stage...", expanded=True)
        
        # 1. Retrieval
        status_box.write("Stage 1: Retrieval & learned reranking...")
        t_ret = time.perf_counter()
        ret_val = engine.retrieve(question, top_k=top_k, use_learned_ranker=use_learned)
        ret_tables = ret_val.get("retrieved_tables", [])
        scores = ret_val.get("scores", [])
        confidence = ret_val.get("confidence", 0.5)
        ret_ms = (time.perf_counter() - t_ret) * 1000
        
        status_box.write(f"✓ Retrieved tables: `{', '.join(ret_tables)}` (Confidence: {confidence:.2%})")
        
        # 2. Generation
        status_box.write("Stage 2: LLM SQL generation...")
        t_gen = time.perf_counter()
        sql_query = generate_sql_query(question, ret_tables)
        gen_ms = (time.perf_counter() - t_gen) * 1000
        status_box.write("✓ SQL query generated.")
        
        # 3. Validation
        status_box.write("Stage 3: AST and schema validation...")
        t_val = time.perf_counter()
        is_valid_syntax, parsing_errors = validate_sql(sql_query)
        val_ms = (time.perf_counter() - t_val) * 1000
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        if is_valid_syntax:
            status_box.write("✓ SQL Syntax is valid!")
        else:
            status_box.write(f"⚠ SQL Validation Errors: {parsing_errors}")
            
        status_box.update(label=f"Pipeline complete in {total_ms:.1f}ms!", state="complete" if is_valid_syntax else "error")
        
        # Log to global metrics
        metrics.record(RequestRecord(
            endpoint="/generate-sql",
            timestamp=time.time(),
            latency_ms=total_ms,
            success=True,
            num_tables_retrieved=len(ret_tables),
            top_confidence=confidence,
            used_learned_ranker=use_learned and "learned" in ret_val.get("model_used", ""),
            retrieval_ms=ret_ms,
            generation_ms=gen_ms,
            validation_ms=val_ms,
        ))
        
        # Log to JSONL
        pipeline_logger.log(
            question=question,
            retrieved_tables=ret_tables,
            scores=scores,
            confidence=confidence,
            model_used=ret_val.get("model_used", "heuristic"),
            sql_generated=sql_query,
            is_valid=is_valid_syntax,
            parsing_errors=parsing_errors,
            latency_ms=total_ms,
            latency_breakdown={
                "retrieval_ms": round(ret_ms, 2),
                "generation_ms": round(gen_ms, 2),
                "validation_ms": round(val_ms, 2),
            }
        )
        
        # Display Results
        res_col1, res_col2 = st.columns([1, 1])
        with res_col1:
            st.markdown("### Generated SQL")
            st.code(sql_query, language="sql")
            
            # Explain SQL Plan
            if st.checkbox("Show SQL Execution Plan (EXPLAIN)"):
                try:
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute(f"EXPLAIN QUERY PLAN {sql_query.strip().rstrip(';')}")
                    explain_rows = cursor.fetchall()
                    conn.close()
                    if explain_rows:
                        st.dataframe(pd.DataFrame(explain_rows))
                except Exception as e:
                    st.error(f"Failed to explain query plan: {e}")
                    
        with res_col2:
            st.markdown("### Execution Results")
            try:
                conn = get_db_connection()
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
                st.dataframe(df, use_container_width=True)
                st.success(f"Returned {len(df)} rows")
            except Exception as e:
                st.error(f"Database execution error: {e}")

# ==========================================
# TAB 2: SCHEMA EXPLORER
# ==========================================
with tabs[1]:
    st.subheader("Database Schema Explorer")
    
    # Simple search bar
    search_query = st.text_input("Search tables or columns", value="")
    
    # Filter schemas
    filtered_tables = []
    for table_name, desc in schema_data.items():
        if not search_query or search_query.lower() in table_name.lower() or search_query.lower() in desc.lower():
            filtered_tables.append(table_name)
            
    if not filtered_tables:
        st.warning("No matching tables found.")
    else:
        selected_table = st.selectbox("Select Table to Explore", options=sorted(filtered_tables))
        
        if selected_table:
            st.markdown(f"## Table: `{selected_table}`")
            
            desc = schema_data[selected_table]
            st.write(desc)
            
            # Fetch actual columns from SQLite DB
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info('{selected_table}')")
                cols = cursor.fetchall()
                conn.close()
                
                cols_df = pd.DataFrame(cols)[["name", "type", "notnull", "dflt_value", "pk"]]
                st.dataframe(cols_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load columns from database: {e}")

# ==========================================
# TAB 3: EVALUATION & BENCHMARK
# ==========================================
with tabs[2]:
    st.subheader("Pipeline Benchmark Evaluation")
    st.write("Run the offline accuracy evaluation on the Beaver dataset subset (25 queries).")
    
    if st.button("Start Benchmark Run", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            query_ds = load_beaver_queries(split="dw")
            eval_queries = [row for row in query_ds if row.get("question") and row.get("sql")][:25]
            
            total_evaluated = 0
            total_recall_5 = 0.0
            total_recall_10 = 0.0
            exact_matches = 0
            execution_matches = 0
            parsing_successes = 0
            total_latency_ms = 0.0
            
            conn = get_db_connection()
            
            for idx, row in enumerate(eval_queries):
                status_text.text(f"Evaluating query #{idx+1} / {len(eval_queries)}: {row['question'][:60]}...")
                
                question = row["question"]
                gold_sql = row["sql"]
                
                # Evaluate Recall
                raw_parsed_tables, ctes = extract_tables_and_ctes(gold_sql)
                tables_no_ctes = raw_parsed_tables - ctes
                gold_tables = {t for t in tables_no_ctes if t in valid_schema_tables}
                
                if not gold_tables:
                    continue
                    
                total_evaluated += 1
                start_time = time.perf_counter()
                
                # Retrieve
                ret_val_5 = engine.retrieve(question, top_k=5)
                ret_tables_5 = [t.upper() for t in ret_val_5.get("retrieved_tables", [])]
                ret_val_10 = engine.retrieve(question, top_k=10)
                ret_tables_10 = [t.upper() for t in ret_val_10.get("retrieved_tables", [])]
                
                # Recall
                matched_5 = gold_tables.intersection(set(ret_tables_5))
                total_recall_5 += len(matched_5) / len(gold_tables)
                
                matched_10 = gold_tables.intersection(set(ret_tables_10))
                total_recall_10 += len(matched_10) / len(gold_tables)
                
                # Generate & Validate
                try:
                    gen_query = generate_sql_query(question, ret_val_5.get("retrieved_tables", []))
                    is_syntax_ok, _ = validate_sql(gen_query)
                    if is_syntax_ok:
                        parsing_successes += 1
                        
                    norm_gen = "".join(gen_query.lower().split())
                    norm_gold = "".join(gold_sql.lower().split())
                    if norm_gen == norm_gold:
                        exact_matches += 1
                        
                    # Execution
                    cursor = conn.cursor()
                    cursor.execute(gold_sql)
                    gold_rows = cursor.fetchall()
                    cursor.execute(gen_query)
                    gen_rows = cursor.fetchall()
                    
                    if set(gen_rows) == set(gold_rows):
                        execution_matches += 1
                except Exception:
                    pass
                    
                total_latency_ms += (time.perf_counter() - start_time) * 1000
                progress_bar.progress((idx + 1) / len(eval_queries))
                
            conn.close()
            status_text.text("Benchmark complete!")
            
            # Show summary metrics
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Recall@5", f"{(total_recall_5 / total_evaluated):.1%}")
            m_col2.metric("Recall@10", f"{(total_recall_10 / total_evaluated):.1%}")
            m_col3.metric("Execution Match Rate", f"{(execution_matches / total_evaluated):.1%}")
            m_col4.metric("Avg Latency", f"{(total_latency_ms / total_evaluated):.0f} ms")
            
        except Exception as e:
            st.error(f"Benchmark run failed: {e}")

# ==========================================
# TAB 4: ML EXPERIMENTS
# ==========================================
with tabs[3]:
    st.subheader("LTR Reranking Models & Experiments")
    
    from app.ml.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()
    runs = tracker.list_runs()
    
    if not runs:
        st.warning("No historical LTR runs logged yet. Run LTR training scripts offline to log runs.")
    else:
        run_options = {f"{r.run_id} ({r.model_type})": r for r in runs}
        selected_run_id = st.selectbox("Select Experiment Run", options=list(run_options.keys()))
        run = run_options[selected_run_id]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Run Configurations")
            st.write(f"**Model Type**: `{run.model_type}`")
            st.write(f"**Timestamp**: {run.timestamp}")
            st.write(f"**Training Samples**: {run.num_training_samples}")
            st.write(f"**CV Folds**: {run.cv_folds}")
            st.write(f"**Training Duration**: {run.training_duration_seconds:.2f} seconds")
            
            st.markdown("### Validation Metrics")
            st.dataframe(pd.DataFrame([run.val_metrics]).T.rename(columns={0: "Score"}))
            
        with col2:
            st.markdown("### Feature Importance")
            if run.feature_importances:
                feat_df = pd.DataFrame(list(run.feature_importances.items()), columns=["Feature", "Importance"])
                feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(15)
                
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax, palette="Blues_d")
                ax.set_title("Top Feature Importances")
                st.pyplot(fig)
            else:
                st.info("No feature importance logged for this model.")

# ==========================================
# TAB 5: SYSTEM ANALYTICS
# ==========================================
with tabs[4]:
    st.subheader("System Performance & Analytics")
    
    metrics_summary = metrics.get_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", metrics_summary["total_requests"])
    col2.metric("Pipeline Error Rate", f"{metrics_summary['error_rate']:.2%}")
    col3.metric("Avg Retrieval Confidence", f"{metrics_summary['avg_confidence']:.2%}")
    col4.metric("LTR Ranker Usage", f"{metrics_summary['learned_ranker_usage_pct']:.2%}")
    
    st.markdown("---")
    
    # Latency percentiles
    st.markdown("### Endpoint Latencies")
    endpoints = metrics_summary.get("endpoints", {})
    if not endpoints:
        st.info("No requests logged yet. Execute query pipelines to populate metrics.")
    else:
        st.dataframe(pd.DataFrame(endpoints).T)
        
        # Pipeline Breakdown
        st.markdown("### Pipeline Phase Breakdown (Avg ms)")
        breakdown = metrics_summary.get("pipeline_breakdown", {})
        if breakdown:
            fig, ax = plt.subplots(figsize=(8, 3.5))
            phases = list(breakdown.keys())
            times = list(breakdown.values())
            sns.barplot(x=times, y=phases, ax=ax, palette="viridis")
            ax.set_xlabel("Time (ms)")
            st.pyplot(fig)

# ==========================================
# TAB 6: PIPELINE LOGS
# ==========================================
with tabs[5]:
    st.subheader("Structured JSONL Audit Logs")
    
    limit = st.slider("Max logs to display", min_value=10, max_value=100, value=50)
    recent_logs = pipeline_logger.get_recent_logs(limit=limit)
    
    if not recent_logs:
        st.info("No logs found. Try executing a query in the workspace first.")
    else:
        logs_df = pd.DataFrame(recent_logs)
        st.dataframe(logs_df, use_container_width=True)
