# Enterprise Text-to-SQL Engine

A production-grade ML pipeline that translates natural language questions into executable SQL queries against large-scale enterprise databases. Built on the Beaver Database Benchmark (97 tables, 5,787 queries) using a multi-stage retrieval-augmented generation (RAG) architecture with classical ML reranking.

Key Results: 86% Retrieval Recall@5, 97% Recall@10, Sub-700ms latency, Learned LTR reranking with LightGBM/XGBoost.

- Python 3.9+
- FastAPI 0.115
- scikit-learn 1.3+
- LightGBM 4.0+
- Docker Ready

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [ML Methodology](#ml-methodology)
- [System Architecture](#system-architecture)
- [Feature Engineering](#feature-engineering)
- [Model Training & Evaluation](#model-training--evaluation)
- [Interactive Dashboard](#interactive-dashboard)
- [Tech Stack & Performance](#tech-stack--performance)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Production Deployment](#production-deployment)
- [Insights & Key Learnings](#insights--key-learnings)
- [License](#license)

---

## Problem Statement

Enterprise databases like the Beaver benchmark contain 97 tables with hundreds of columns. Translating natural language questions to SQL requires identifying the correct 3 to 5 tables from this massive schema space. Naive approaches (pasting all schemas into an LLM prompt) fail due to:

- **Token overflow**: 97 tables with 20+ columns exceeds context limits, preventing the LLM from processing the full schema.
- **Hallucination**: Too many similar table names confuse the model, causing it to invent non-existent columns, tables, and joins.
- **Latency & cost**: Processing 50K+ tokens per query is 5x to 10x slower and highly expensive.

The core insight is that if we can identify the correct tables in the top-5, downstream SQL generation succeeds. If we can't, the downstream generator has a 0% chance of success. This makes the system fundamentally a schema retrieval and ranking problem rather than just an NLP problem.

---

## ML Methodology

### 4-Stage Funnel Architecture

The system is designed as a 4-stage funnel that progressively narrows down the search space:

1. **Query Expansion**: Adds synonyms, related keywords, and SQL-specific keywords to bridge the vocabulary gap.
2. **Candidate Retrieval (Broad Net)**: Narrows the search space from 97 tables to 25 candidate tables using a hybrid BM25 and vector similarity search.
3. **Precision Reranking (Fine Filter)**: Applies cross-encoder scoring and a learned LTR model using 28 features to score and select the top 5 tables.
4. **SQL Generation**: Generates the final executable SQL query via LLM using only the schema context of the top 5 tables.

### Classical ML Reranking (LTR Upgrade)

The heuristic reranking rules (consisting of hand-coded conditional statements) have been replaced with a learned-to-rank (LTR) model using gradient boosting:

- **Training data**: Generated from gold SQL queries by extracting correct tables as positive labels (1) and incorrect tables as negative labels (0). This produces around 1,250 training samples (50 queries x 25 candidates).
- **Primary model**: LightGBM, selected for its fast training and high performance on small tabular datasets.
- **Alternative models**: XGBoost, Random Forest, and Gradient Boosting Classifier are supported for comparison.
- **Evaluation strategy**: Grouped K-Fold cross-validation (K=5) grouped by question to prevent data leakage between candidate queries.

---

## System Architecture

### End-to-End Pipeline Flow

```
User Question
    |
    v
Query Expansion (Rule-based + LLM)
    |
    +-----> BM25 Keyword Search ------+
    |                                 v
    +-----> Cosine Vector Search ---> Hybrid Merge (alpha=0.5 fusion)
                                      |
                                      v
                               Name/Category Boosts + Graph Score Propagation
                                      |
                                      v
                               Cross-Encoder Reranking (MiniLM-L-6-v2)
                                      |
                                      v
                               Feature Extraction (28 features per table)
                                      |
                                      v
                               Learned Ranker (LightGBM/XGBoost)
                                      |
                                      v
                               Top 5 Tables Selected
                                      |
                                      v
                               SQL Generation (Llama-3.1-8B via Groq)
                                      |
                                      v
                               SQL Validation (sqlglot AST + EXPLAIN)
                                      |
                                      v
                               JSON Response to Client
```

### Database Integration

The SQL execution validation engine is built directly on SQLite with custom aggregate functions registered at database initialization:
- **Variance**: Computes statistical variance.
- **Standard Deviation**: Computes standard deviation.
- **Log**: Computes natural logarithm.

These custom functions bridge the gap between production SQL statements and SQLite's default capabilities.

---

## Feature Engineering

The learned ranker uses 28 handcrafted features across 5 categories to analyze (question, table) candidate pairs:

1. **Lexical Features**: BM25 score, TF-IDF cosine, Jaccard word overlap, character 3-gram overlap, exact name match, partial name match, and token overlap.
2. **Semantic Features**: Bi-encoder cosine similarity, cross-encoder raw logits, and calibrated cross-encoder probability scores.
3. **Structural Features**: Column count, foreign key relations count, schema graph degree, join key overlap with candidates, and count of shared join keys.
4. **Query Features**: Length of query (words and characters), aggregation keyword presence, join keyword presence, subquery indicator presence, and count of detected SQL keywords.
5. **Interaction Features**: Product of BM25 and cosine scores, BM25 rank position, cosine similarity rank position, rank difference, score ratio, LLM expansion overlap, and category prefix match.

---

## Model Training & Evaluation

### Training Commands

You can train and evaluate LTR models using the command line:

```bash
# Train and compare all model types (LightGBM, XGBoost, Random Forest)
python -m app.ml.train_ranker --model all

# Train a specific model type
python -m app.ml.train_ranker --model lightgbm --cv-folds 5

# Customize training bounds
python -m app.ml.train_ranker --model xgboost --max-queries 50 --candidates 25
```

### Accuracy Evaluation

Evaluate schema retrieval accuracy against the Beaver gold standard:

```bash
python -m scripts.test_retrieval_accuracy
```

---

## Interactive Dashboard

The project serves a web application served directly from the FastAPI backend at port 8001:

- **Query**: Natural language input to run SQL generation with real-time pipeline execution progress blocks.
- **Benchmark**: Evaluates retrieval recall (Recall@5, Recall@10), query syntax validity, and average latency.
- **Schema Explorer**: Searchable interface to explore all 97 tables, columns, and foreign-key links.
- **Experiments**: Visualizes trained LTR runs and feature importance coefficients.
- **Metrics**: Production analytics dashboard monitoring latency percentiles, error rates, and stage-by-stage timings.

---

## Tech Stack & Performance

- **FastAPI**: Backend web framework.
- **Vector Embeddings**: BAAI/bge-small-en-v1.5 bi-encoder.
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2.
- **LLM**: Llama-3.1-8B-Instant via Groq API.
- **Database**: SQLite beaver_dw.db with custom registered functions (VARIANCE, STDDEV, LOG).
- **ML Frameworks**: scikit-learn, LightGBM, XGBoost.
- **Validation**: sqlglot AST syntax checking and EXPLAIN QUERY PLAN verification.
- **Frontend**: Vite, React, TypeScript, and Zustand (Proxy forwarded to port 8001).

---

## Repository Structure

```text
text-to-sql/
├── app/
│   ├── main.py                          # FastAPI application entrypoint
│   ├── core/
│   │   ├── config.py                    # App configuration and directories
│   │   ├── logging.py                   # Logger setup
│   │   ├── metrics_collector.py         # Telemetry statistics collector
│   │   └── pipeline_logger.py           # Structured JSONL audit logs
│   ├── models/
│   │   ├── requests.py                  # API request schemas
│   │   └── responses.py                 # API response schemas
│   ├── retrieval/
│   │   ├── engine.py                    # Multi-stage retrieval and learned ranker
│   │   └── schema_loader.py             # Schema loading and relationship graphs
│   ├── generation/
│   │   └── generator.py                 # LLM generation and Groq API calls
│   ├── database/
│   │   ├── connection.py                # SQLite connection and custom aggregation
│   │   └── validator.py                 # SQL statement validation checks
│   ├── ml/                              # LTR machine learning codebase
│   │   ├── feature_engineering.py       # Feature extraction matrix generator
│   │   ├── learned_ranker.py            # LightGBM/XGBoost training pipeline
│   │   ├── experiment_tracker.py        # Experiment tracking logs
│   │   └── train_ranker.py              # CLI ranker training script
│   └── static/                          # React compiled bundle directory
│       ├── index.html                   # HTML entrypoint
│       └── assets/                      # Production CSS and JS assets
├── database/
│   └── beaver_dw.db                     # SQLite beaver database
├── models/                              # Serialized LTR model artifacts
│   └── ranker_v1.joblib                 # Serialized LightGBM ranker
├── scripts/
│   └── test_retrieval_accuracy.py       # Accuracy evaluation runner
├── Dockerfile                           # Multi-stage production container
├── docker-compose.yml                   # Container orchestration config
├── requirements.txt                     # Python packages list
├── explanation.md                       # Comprehensive design explanation
└── README.md                            # Project documentation
```

---

## Quick Start

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/text-to-sql.git && cd text-to-sql
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and supply your HF_TOKEN and GROQ_API_KEY
   ```

4. **Start the FastAPI backend server**:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
   ```

5. **Open the web dashboard**:
   Navigate to `http://localhost:8001` in your browser.

---

## API Reference

- **GET /**: Serves the web dashboard page.
- **POST /retrieve**: Returns top relevant tables matching a question.
  - Payload: `{"question": "Which departments have more than 100 students?", "top_k": 5}`
- **POST /generate-sql**: Full RAG pipeline execution returning generated SQL, source tables, confidence, and latencies.
  - Payload: `{"question": "Which departments have more than 100 students?"}`
- **POST /benchmark**: Evaluates recall and query execution rates over benchmark datasets.
- **GET /health**: Validates model initialization state and database connection status.
- **GET /api/schema**: Returns Beaver dataset tables list with fields and relationship constraints.
- **GET /api/experiments**: Returns historical cross-validation scores.
- **GET /api/metrics**: Returns average endpoint latency percentiles and error metrics.
- **GET /docs**: Interactive Swagger OpenAPI documentation.

---

## Production Deployment

### Production Deployment Steps

Follow these instructions to deploy the engine in a production-ready environment.

#### Method A: Multi-Stage Docker Container (Recommended)

1. **Ensure your environment file (.env) is fully configured**:
   Supply valid values for `HF_TOKEN` and `GROQ_API_KEY`.

2. **Build the Docker container image**:
   ```bash
   docker build -t text-to-sql-api .
   ```

3. **Run the container**:
   ```bash
   docker run -d -p 8001:8001 --env-file .env --name text-to-sql-service text-to-sql-api
   ```
   This command starts the API on port 8001 mapping to the internal port 8001 of the runtime.

4. **Validate the deployment**:
   Run the health check endpoint:
   ```bash
   curl http://localhost:8001/health
   ```

#### Method B: Container Orchestration via Docker Compose

1. **Start the services in detached mode**:
   ```bash
   docker-compose up -d --build
   ```
   This maps the backend database and model artifacts directory to local volumes, persisting model training files and custom SQLite records.

2. **Inspect the logs**:
   ```bash
   docker-compose logs -f
   ```

3. **Shutdown services**:
   ```bash
   docker-compose down
   ```

---

## Insights & Key Learnings

1. **Schema retrieval is the bottleneck, not SQL generation**: Correct table identification determines 100% of downstream success. If the correct tables are in the top-5, the LLM will generate correct queries.
2. **Hybrid search is superior to vector or keyword search alone**: BM25 catches exact keyword matches; vector similarity catches conceptual meanings. Melding both covers query search failures.
3. **Cross-encoders improve reranking quality**: Rerankers score candidates jointly, capturing fine interactions that bi-encoders miss.
4. **Heuristic rules do not generalize**: Hundreds of lines of hand-coded boosts fail on new queries. Machine learning models train and optimize over the full data distribution.
5. **CTE alias filtering is critical for accurate evaluation metrics**: Removing CTE aliases from correct labels prevents recall scores from being artificially inflated.

## Future Improvements

We plan to implement the following core updates to mature the Text-to-SQL architecture:

1. **Multi-Turn Conversational SQL Generation**: Implement chat session memory and tracking so the retrieval funnel and SQL generator can interpret references to previous query results (e.g. "filter the last results to only active students").
2. **Self-Correction Agent Loops**: Design an agentic feedback cycle that executes generated SQL inside a dry-run transaction. If SQLite raises syntax/execution exceptions or if the AST structure fails validations, the error stack trace is fed back to the LLM for automatic query correction (up to 3 retries).
3. **Graph Neural Network (GNN) Schema Ranking**: Upgrade the LTR feature ranker with GNN architectures (e.g. GraphSAGE/GCN) to model tables as nodes and foreign key constraints as edges. This enables joint path retrieval rather than independent table ranking.
4. **Active Schema Pruning & Context Pruning**: Implement graph path reachability filters during Stage 2 retrieval to prune tables that have no valid join paths to the existing candidate set, maximizing precision.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
