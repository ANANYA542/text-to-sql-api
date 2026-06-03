from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

class TableDetail(BaseModel):
    relevance_score: float
    reason: str

class RetrieveResponse(BaseModel):
    retrieved_tables: list[str]
    scores: list[float]
    confidence: float
    details: dict[str, TableDetail]

class GenerateSQLResponse(BaseModel):
    sql: str
    retrieved_tables: list[str]
    is_valid_syntax: bool
    parsing_errors: Optional[str] = None
    confidence: float
    prompt_used: str

class BenchmarkMetrics(BaseModel):
    retrieval_recall_at_5: float
    retrieval_recall_at_10: float
    sql_exact_match_accuracy: float
    sql_execution_match_accuracy: float
    parsing_success_rate: float
    average_latency_ms: float

class BenchmarkSubtaskBreakdown(BaseModel):
    multi_table_retrieval: float
    column_mapping: float
    join_detection: float
    domain_knowledge: float

class BenchmarkErrorAnalysis(BaseModel):
    retrieval_failures: int
    parsing_failures: int
    execution_failures: int
    logic_errors: int

class BenchmarkResponse(BaseModel):
    total_queries: int
    metrics: BenchmarkMetrics
    subtask_breakdown: BenchmarkSubtaskBreakdown
    error_analysis: BenchmarkErrorAnalysis

class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
