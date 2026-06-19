"""
Structured Pipeline Logger for ML audit and debugging.

Logs each request through the pipeline as a structured JSON record,
capturing every intermediate result for offline analysis.

Design:
    - File-based JSON Lines (JSONL) format — one JSON object per line.
    - JSONL is append-only, so concurrent writes don't corrupt the file.
    - Rotates to a new file when the current one exceeds MAX_FILE_SIZE.
    - No external dependencies (no ELK stack, no cloud logging).

Why JSONL over plain text logs?
    - JSONL is machine-parseable: you can `jq` filter, pandas.read_json, etc.
    - Plain text logs require regex parsing which is fragile.
    - One object per line means partial writes don't corrupt the whole file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default log directory
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[2] / ".cache" / "pipeline_logs"

# Rotate when file exceeds 10MB
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB


class PipelineLogger:
    """
    Logs structured pipeline execution records to JSONL files.

    Usage:
        pl = PipelineLogger()
        pl.log(
            question="Which departments have more than 100 students?",
            retrieved_tables=["SIS_DEPARTMENT", "STUDENT_DEPARTMENT"],
            scores=[0.95, 0.87],
            model_used="lightgbm",
            latency_ms=320.5,
            sql_generated="SELECT ...",
            is_valid=True,
        )
    """

    def __init__(self, log_dir: Path | str | None = None):
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_file: Path | None = None
        self._ensure_log_file()
        logger.info(f"PipelineLogger initialized. Log dir: {self.log_dir}")

    def _ensure_log_file(self):
        """Creates or rotates the log file if needed."""
        if self._current_file and self._current_file.exists():
            if self._current_file.stat().st_size < MAX_FILE_SIZE:
                return  # Current file is fine

        # Create a new log file with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_file = self.log_dir / f"pipeline_{timestamp}.jsonl"

    def log(
        self,
        question: str,
        retrieved_tables: list[str] | None = None,
        scores: list[float] | None = None,
        confidence: float = 0.0,
        model_used: str = "heuristic",
        sql_generated: str | None = None,
        is_valid: bool | None = None,
        parsing_errors: str | None = None,
        latency_ms: float = 0.0,
        latency_breakdown: dict[str, float] | None = None,
        feature_importances: dict[str, float] | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ):
        """
        Logs a single pipeline execution record.

        All parameters are optional except question — log whatever you have.
        """
        self._ensure_log_file()

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "retrieved_tables": retrieved_tables or [],
            "scores": scores or [],
            "confidence": confidence,
            "model_used": model_used,
            "sql_generated": sql_generated,
            "is_valid": is_valid,
            "parsing_errors": parsing_errors,
            "latency_ms": round(latency_ms, 2),
            "latency_breakdown": latency_breakdown or {},
            "error": error,
        }

        if feature_importances:
            # Only log top 5 feature importances to save space
            top_5 = dict(list(feature_importances.items())[:5])
            record["top_feature_importances"] = top_5

        if extra:
            record["extra"] = extra

        try:
            with open(self._current_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write pipeline log: {e}")

    def get_recent_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Returns the most recent pipeline log entries.

        Reads from the current log file in reverse order.
        """
        if not self._current_file or not self._current_file.exists():
            return []

        try:
            with open(self._current_file, "r") as f:
                lines = f.readlines()

            records = []
            for line in reversed(lines[-limit:]):
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

            return records
        except Exception as e:
            logger.error(f"Failed to read pipeline logs: {e}")
            return []
