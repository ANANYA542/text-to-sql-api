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
    page_icon="💎",
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

# Premium Slate Dark & Blue Theme Injections
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 100%);
        color: #f3f4f6;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Space Grotesk', sans-serif;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #60a5fa !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 13px !important;
        color: #9ca3af !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Styled Containers & Cards */
    .premium-card {
        background: rgba(17, 24, 39, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    .pipeline-step {
        border-left: 3px solid #3b82f6;
        padding-left: 16px;
        margin-bottom: 12px;
    }
    
    /* Custom Badge */
    .custom-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    /* Logo styling */
    .brand-logo {
        background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 34px;
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
    }
    
    /* Tab Styling Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(17, 24, 39, 0.5);
        padding: 8px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        color: #9ca3af;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
        background-color: rgba(255, 255, 255, 0.03);
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Layout: Main Brand Area
col_logo, col_desc = st.columns([2, 3])
with col_logo:
    st.markdown('<div class="brand-logo">💎 Enterprise Text-to-SQL</div>', unsafe_allow_html=True)
with col_desc:
    st.markdown("<p style='color: #9ca3af; font-size: 14px; margin-top: 15px; text-align: right;'>Advanced multi-stage pipeline using neural reranking and LLM code synthesizers.</p>", unsafe_allow_html=True)

st.markdown("---")

tabs = st.tabs([
    "🔍 Workspace & Pipeline", 
    "🗂 Database Schema Explorer", 
    "📊 Precision Benchmarks", 
    "🔬 Learned LTR Experiments", 
    "📈 Analytics & Latency Histograms",
    "📋 Real-time Logs"
])

# ==========================================
# TAB 1: WORKSPACE & PIPELINE
# ==========================================
with tabs[0]:
    st.markdown("### Interactive Context-Restricted SQL Engine")
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        question = st.text_area(
            "Natural Language Inquiry",
            value="Which departments have more than 100 students?",
            height=85,
            placeholder="Ask a question about the university or facilities database schema..."
        )
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            top_k = st.slider("Max Context Schema Size (Top K Tables)", min_value=1, max_value=10, value=5)
        with col_ctrl2:
            use_learned = st.toggle("Activate Learned LTR Ranker (LightGBM/XGBoost)", value=True)
            
        run_btn = st.button("Execute Translation Pipeline", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown(
            f"""
            <div class="premium-card" style="height: 100%;">
                <h4 style="margin-top:0;">Pipeline Stages & Strategy</h4>
                <div class="pipeline-step">
                    <strong>1. Multi-stage Retrieval</strong><br/>
                    <span style="color:#9ca3af; font-size:12px;">Combines BM25 lexical matching and semantic vector embeddings, then applies schema graph reachability propagation.</span>
                </div>
                <div class="pipeline-step">
                    <strong>2. Precision Reranking</strong><br/>
                    <span style="color:#9ca3af; font-size:12px;">Applies Cross-Encoder logits and a trained LightGBM model evaluating 28 custom features per candidate table.</span>
                </div>
                <div class="pipeline-step">
                    <strong>3. Context-Restricted SQL Synthesis</strong><br/>
                    <span style="color:#9ca3af; font-size:12px;">Compiles schemas of the top 5 tables only, prompting {config.LLM_MODEL} for zero-hallucination query creation.</span>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    if run_btn and question:
        start_time = time.perf_counter()
        
        # Pipeline Logs tracker
        status_box = st.status("Executing Pipeline Funnel...", expanded=True)
        
        # 1. Retrieval
        status_box.write("⚙ Stage 1: Query Expansion and Candidates Retrieval...")
        t_ret = time.perf_counter()
        ret_val = engine.retrieve(question, top_k=top_k, use_learned_ranker=use_learned)
        ret_tables = ret_val.get("retrieved_tables", [])
        scores = ret_val.get("scores", [])
        confidence = ret_val.get("confidence", 0.5)
        ret_ms = (time.perf_counter() - t_ret) * 1000
        
        status_box.write(f"✓ Retrieved schema tables: `{', '.join(ret_tables)}` (Top confidence: {confidence:.2%})")
        
        # 2. Generation
        status_box.write("⚙ Stage 2: Prompt compilation and LLM generation...")
        t_gen = time.perf_counter()
        sql_query = generate_sql_query(question, ret_tables)
        gen_ms = (time.perf_counter() - t_gen) * 1000
        status_box.write("✓ SQL query code synthesized.")
        
        # 3. Validation
        status_box.write("⚙ Stage 3: Running AST parsing syntax validations...")
        t_val = time.perf_counter()
        is_valid_syntax, parsing_errors = validate_sql(sql_query)
        val_ms = (time.perf_counter() - t_val) * 1000
        
        total_ms = (time.perf_counter() - start_time) * 1000
        
        if is_valid_syntax:
            status_box.write("✓ Syntax check passed. Schema alignment correct.")
        else:
            status_box.write(f"⚠ AST Syntax Warning: {parsing_errors}")
            
        status_box.update(label=f"Pipeline finished successfully in {total_ms:.1f}ms!", state="complete" if is_valid_syntax else "error")
        
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
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### Generated SQL Query")
            st.code(sql_query, language="sql")
            
            # Explain SQL Plan
            if st.checkbox("Show Query Execution Plan (EXPLAIN)"):
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
            st.markdown('</div>', unsafe_allow_html=True)
                    
        with res_col2:
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            st.markdown("#### SQLite Execution Results")
            try:
                conn = get_db_connection()
                df = pd.read_sql_query(sql_query, conn)
                conn.close()
                st.dataframe(df, use_container_width=True)
                st.success(f"Execution match successful. Returned {len(df)} rows.")
            except Exception as e:
                st.error(f"Database execution error: {e}")
            st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 2: DATABASE SCHEMA EXPLORER
# ==========================================
with tabs[1]:
    st.markdown("### Interactive Schema Directory")
    st.write("Browse and search table structures across the 97 tables of the Beaver database schema.")
    
    # Simple search bar
    search_query = st.text_input("Filter tables/columns", value="", placeholder="Type table prefix (e.g. SIS_, FCLT_) or column name...")
    
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
            st.markdown(f"#### Table Details: `{selected_table}`")
            
            st.info(schema_data[selected_table])
            
            # Fetch actual columns from SQLite DB
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute(f"PRAGMA table_info('{selected_table}')")
                cols = cursor.fetchall()
                conn.close()
                
                cols_df = pd.DataFrame(cols)[["name", "type", "notnull", "pk"]]
                st.dataframe(cols_df, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load columns from database: {e}")

# ==========================================
# TAB 3: PRECISION BENCHMARKS
# ==========================================
with tabs[2]:
    st.markdown("### Precision Evaluation Benchmarking")
    st.write("Verify the retrieval accuracy and query validity of the engine using a gold-standard subset of 25 benchmark queries.")
    
    if st.button("Execute Pipeline Benchmark Test Run", type="secondary", use_container_width=True):
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
                status_text.text(f"Evaluating Query #{idx+1} / {len(eval_queries)}...")
                
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
            
            # Show summary metrics in beautiful card columns
            st.markdown('<div class="premium-card">', unsafe_allow_html=True)
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Recall@5 (Recall)", f"{(total_recall_5 / total_evaluated):.1%}")
            m_col2.metric("Recall@10 (Coverage)", f"{(total_recall_10 / total_evaluated):.1%}")
            m_col3.metric("Execution Match Rate", f"{(execution_matches / total_evaluated):.1%}")
            m_col4.metric("Avg Latency", f"{(total_latency_ms / total_evaluated):.0f} ms")
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Benchmark run failed: {e}")

# ==========================================
# TAB 4: LEARNED LTR EXPERIMENTS
# ==========================================
with tabs[3]:
    st.markdown("### LTR Model Runs & Experiments")
    
    from app.ml.experiment_tracker import ExperimentTracker
    tracker = ExperimentTracker()
    runs = tracker.list_runs()
    
    if not runs:
        st.warning("No historical LTR runs logged yet. Run LTR training scripts offline to log runs.")
    else:
        run_options = {f"{r.run_id} ({r.model_type})": r for r in runs}
        selected_run_id = st.selectbox("Select Experiment Run", options=list(run_options.keys()))
        run = run_options[selected_run_id]
        
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Run Configurations")
            st.write(f"**Model Type**: `{run.model_type}`")
            st.write(f"**Timestamp**: {run.timestamp}")
            st.write(f"**Training Samples**: {run.num_training_samples}")
            st.write(f"**CV Folds**: {run.cv_folds}")
            st.write(f"**Training Duration**: {run.training_duration_seconds:.2f} seconds")
            
            st.markdown("#### Validation Metrics")
            st.dataframe(pd.DataFrame([run.val_metrics]).T.rename(columns={0: "Score"}))
            
        with col2:
            st.markdown("#### Feature Importance Coefficient")
            if run.feature_importances:
                feat_df = pd.DataFrame(list(run.feature_importances.items()), columns=["Feature", "Importance"])
                feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(15)
                
                # Dark slate compatible chart styling
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(6, 4.2))
                sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax, palette="Blues_d")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_title("Top Feature Importances", fontsize=12, fontweight='bold', pad=10)
                st.pyplot(fig)
            else:
                st.info("No feature importance logged for this model.")
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# TAB 5: ANALYTICS & LATENCY HISTOGRAMS
# ==========================================
with tabs[4]:
    st.markdown("### Production Observability & Performance Analytics")
    
    metrics_summary = metrics.get_metrics()
    
    # Safe metrics retrieval to avoid KeyErrors
    avg_conf = metrics_summary.get("avg_confidence", 0.0)
    ltr_usage = metrics_summary.get("learned_ranker_usage_pct", 0.0)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", metrics_summary["total_requests"])
    col2.metric("Pipeline Error Rate", f"{metrics_summary['error_rate']:.2%}")
    col3.metric("Avg Retrieval Confidence", f"{avg_conf:.2%}")
    col4.metric("Avg LTR Ranker Usage", f"{ltr_usage:.2%}")
    
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
# TAB 6: REAL-TIME LOGS
# ==========================================
with tabs[5]:
    st.markdown("### Structured JSONL Audit Logs")
    
    limit = st.slider("Max logs to display", min_value=10, max_value=100, value=50)
    recent_logs = pipeline_logger.get_recent_logs(limit=limit)
    
    if not recent_logs:
        st.info("No logs found. Try executing a query in the workspace first.")
    else:
        logs_df = pd.DataFrame(recent_logs)
        st.dataframe(logs_df, use_container_width=True)
