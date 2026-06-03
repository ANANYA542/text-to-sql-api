# Enterprise Text-to-SQL API

An enterprise-grade Text-to-SQL retrieval and generation API built for the Beaver benchmark.

## Architecture

```
                  +----------------------------------+
                  |           User Question          |
                  +----------------------------------+
                                    |
                                    v
                  +----------------------------------+
                  |      Schema Context Builder      |
                  |     (5-Stage Retrieval Engine)   |
                  +----------------------------------+
                                    |
                                    v
                  +----------------------------------+
                  |            LLM Layer             | <--- [In Progress]
                  |         (Anthropic Claude)       |
                  +----------------------------------+
                                    |
                                    v
                  +----------------------------------+
                  |         Validation Layer         | <--- [In Progress]
                  |             (sqlglot)            |
                  +----------------------------------+
                                    |
                                    v
                  +----------------------------------+
                  |             DB Layer             | <--- [In Progress]
                  |             (SQLite)             |
                  +----------------------------------+
                                    |
                                    v
                  +----------------------------------+
                  |          Metrics Layer           |
                  +----------------------------------+
                                    |
                                    v
                  +----------------------------------+
                  |         FastAPI Response         |
                  +----------------------------------+
```

## Current Metrics

*   **Retrieval Recall@5:** **72.67%** (measured on the Beaver dataset `dw` split across 50 gold queries).

## Retrieval Pipeline

The engine utilizes a 5-stage retrieval pipeline to resolve relevant schema context:

1.  **Stage 1: Query Expansion:** Translates common natural language indicators (e.g. "average", "most represented", "top") into database concept terms (like `AVG`, `GROUP BY`, `ORDER BY LIMIT`) and infers likely table names based on singular/plural forms.
2.  **Stage 2: Parallel Retrieval:** Searches the database schema concurrently via two paths:
    *   **Lexical:** BM25 keyword search over tokenized table schemas.
    *   **Semantic:** Cosine similarity via bi-encoder embeddings (`BAAI/bge-large-en-v1.5`) over enriched schema descriptions.
3.  **Stage 3: Hybrid Merge:** Combines lexical and semantic retrieval outputs using normalized reciprocal scores to ensure robustness against both exact vocabulary matches and conceptual semantic alignments.
4.  **Stage 4: Cross-Encoder Reranking:** Inputs the query-schema candidate pairs into a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) to capture deep interactive attention representation.
5.  **Stage 5: Metadata Boosting:** A final heuristic booster that guarantees direct mentions of tables in natural language questions are force-included in the final top-k retrieval.

## Dataset

*   **Source:** `beaverbench/beaver-table` and `beaverbench/beaver-query` on HuggingFace.
*   **Split:** Data Warehouse (`dw`) split.
*   **Scope:** 97 physical database tables containing dense columns, types, and descriptions mapping to natural language SQL queries.

## Project Structure

```
text-to-sql/
├── app/
│   ├── main.py                     # FastAPI application entrypoint and routes
│   ├── retrieval/
│   │   ├── engine.py               # Core 5-stage RetrievalEngine implementation
│   │   └── schema_loader.py        # Beaver schema loader and description generator
│   └── models/
│       ├── requests.py             # Request Pydantic models (RetrieveRequest)
│       └── responses.py            # Response Pydantic models (RetrieveResponse)
├── scripts/
│   └── test_retrieval_accuracy.py  # Recall@5 accuracy evaluation suite
├── requirements.txt                # Python package dependencies
└── README.md                       # Documentation
```

## Setup Instructions

### Prerequisites
*   Python 3.11+
*   HuggingFace Account logged in via CLI (`huggingface-cli login`) or `HF_TOKEN` set in environment.

### 1. Clone the repository
```bash
git clone <repo-url>
cd text-to-sql
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Setup
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_write_token
```

### 4. Run the API Server
Start the FastAPI server utilizing Uvicorn:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

### 1. Health Check
Checks model loading and server status.

**Request:**
```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "engine_loaded": true,
  "num_tables": 97
}
```

### 2. Retrieve Schema Context
Retrieves the top `k` most relevant tables for a natural language question.

**Request:**
```bash
curl -X POST http://localhost:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question": "For each academic building at MIT with more than 10 rooms, list the building name and its type.", "top_k": 5}'
```

**Response:**
```json
{
  "retrieved_tables": [
    "FCLT_BUILDING",
    "BUILDINGS",
    "FCLT_ROOMS",
    "FCLT_BUILDING_HIST",
    "SPACE_DETAIL"
  ],
  "scores": [0.8921, 0.7654, 0.6512, 0.5412, 0.4321],
  "confidence": 0.6564,
  "details": {
    "FCLT_BUILDING": {
      "relevance_score": 0.8921,
      "reason": "Semantically relevant to the query (score=0.8921)"
    }
  }
}
```

---

## What's Coming Next (In Progress)

*   `POST /generate-sql`: An LLM-powered SQL generation endpoint using Anthropic's Claude.
*   `POST /benchmark`: Runs end-to-end Text-to-SQL correctness benchmarks on the beaver-query dataset.
*   **SQLite Database Execution**: Executing generated SQL query strings against local SQLite DB instances for validation.
*   **SQL Validation Layer**: Syntactic and semantic validation of SQL ASTs via `sqlglot` prior to database execution.
