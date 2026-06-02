from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import dotenv
dotenv.load_dotenv()

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from models import RetrieveRequest, RetrieveResponse, ErrorResponse
from schema_loader import load_beaver_schema
from retrieval import RetrievalEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# global engine loaded once at startup
engine: RetrievalEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    logger.info("=== Starting up: loading schema and models ===")
    try:
        schema = load_beaver_schema(split="dw")
        engine = RetrievalEngine(schema)
        logger.info("=== Startup complete ===")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
    yield
    logger.info("=== Shutting down ===")


app = FastAPI(
    title="Text-to-SQL Retrieval API",
    description="Retrieves relevant database tables for natural language questions using the Beaver benchmark.",
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
        raise HTTPException(status_code=500, detail="Engine not initialized")

    try:
        result = engine.retrieve(request.question, top_k=5)
        return RetrieveResponse(**result)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "num_tables": len(engine.table_names) if engine else 0,
    }
