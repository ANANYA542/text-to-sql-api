"""
Production Metrics Collector for request-level monitoring.

Tracks per-endpoint latency histograms, retrieval accuracy over time,
prediction confidence distributions, and error rates.

Design:
    - In-memory ring buffer (last 1000 requests) — no database dependency.
    - Thread-safe via collections.deque (GIL-protected for single appends).
    - Exposes summary statistics via get_metrics() for the /api/metrics endpoint.

Why not Prometheus/Grafana?
    For a portfolio project, an external monitoring stack is overkill.
    The in-memory collector gives us the same observability for demos
    without requiring Docker, Prometheus, or Grafana setup.
"""

from __future__ import annotations

import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Maximum number of requests to keep in the ring buffer.
# 1000 requests × ~200 bytes per entry = ~200KB — negligible memory.
MAX_HISTORY = 1000


@dataclass
class RequestRecord:
    """A single request record for monitoring."""
    endpoint: str
    timestamp: float
    latency_ms: float
    success: bool
    error_type: str | None = None

    # Retrieval-specific fields (only for /retrieve and /generate-sql)
    num_tables_retrieved: int = 0
    top_confidence: float = 0.0
    used_learned_ranker: bool = False

    # Pipeline stage breakdown (ms)
    expansion_ms: float = 0.0
    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    generation_ms: float = 0.0
    validation_ms: float = 0.0


class MetricsCollector:
    """
    Collects and summarizes request-level metrics.

    Usage:
        collector = MetricsCollector()

        # Record a request
        record = RequestRecord(
            endpoint="/generate-sql",
            timestamp=time.time(),
            latency_ms=320.5,
            success=True,
            ...
        )
        collector.record(record)

        # Get summary for dashboard
        summary = collector.get_metrics()
    """

    def __init__(self, max_history: int = MAX_HISTORY):
        self._history: deque[RequestRecord] = deque(maxlen=max_history)
        self._total_requests = 0
        self._total_errors = 0
        logger.info(f"MetricsCollector initialized (buffer size: {max_history})")

    def record(self, record: RequestRecord):
        """Records a single request."""
        self._history.append(record)
        self._total_requests += 1
        if not record.success:
            self._total_errors += 1

    def get_metrics(self) -> dict[str, Any]:
        """
        Returns a summary of all collected metrics.

        Includes:
        - Total request count and error rate
        - Per-endpoint latency percentiles (P50, P95, P99)
        - Average confidence score
        - Learned ranker usage rate
        - Pipeline stage latency breakdown
        """
        if not self._history:
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "endpoints": {},
                "pipeline_breakdown": {},
            }

        records = list(self._history)

        # Overall stats
        error_rate = self._total_errors / self._total_requests if self._total_requests > 0 else 0.0

        # Per-endpoint stats
        endpoints: dict[str, dict[str, Any]] = {}
        for endpoint in set(r.endpoint for r in records):
            ep_records = [r for r in records if r.endpoint == endpoint]
            latencies = [r.latency_ms for r in ep_records]

            endpoints[endpoint] = {
                "count": len(ep_records),
                "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1),
                "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1),
                "latency_p99_ms": round(float(np.percentile(latencies, 99)), 1),
                "latency_avg_ms": round(float(np.mean(latencies)), 1),
                "error_count": sum(1 for r in ep_records if not r.success),
            }

        # Retrieval-specific stats
        retrieval_records = [
            r for r in records
            if r.endpoint in ("/retrieve", "/generate-sql") and r.success
        ]

        avg_confidence = 0.0
        learned_ranker_pct = 0.0
        if retrieval_records:
            confidences = [r.top_confidence for r in retrieval_records]
            avg_confidence = round(float(np.mean(confidences)), 4)
            learned_ranker_pct = round(
                sum(1 for r in retrieval_records if r.used_learned_ranker) / len(retrieval_records),
                4,
            )

        # Pipeline breakdown (average ms per stage)
        pipeline_breakdown = {}
        if retrieval_records:
            pipeline_breakdown = {
                "expansion_ms": round(float(np.mean([r.expansion_ms for r in retrieval_records])), 1),
                "retrieval_ms": round(float(np.mean([r.retrieval_ms for r in retrieval_records])), 1),
                "reranking_ms": round(float(np.mean([r.reranking_ms for r in retrieval_records])), 1),
                "generation_ms": round(float(np.mean([r.generation_ms for r in retrieval_records])), 1),
                "validation_ms": round(float(np.mean([r.validation_ms for r in retrieval_records])), 1),
            }

        return {
            "total_requests": self._total_requests,
            "total_errors": self._total_errors,
            "error_rate": round(error_rate, 4),
            "avg_confidence": avg_confidence,
            "learned_ranker_usage_pct": learned_ranker_pct,
            "buffer_size": len(records),
            "endpoints": endpoints,
            "pipeline_breakdown": pipeline_breakdown,
        }

    def get_latency_history(
        self, endpoint: str | None = None, limit: int = 100
    ) -> list[dict[str, float]]:
        """
        Returns recent latency data points for charting.

        Each point contains: timestamp, latency_ms, confidence.
        """
        records = list(self._history)
        if endpoint:
            records = [r for r in records if r.endpoint == endpoint]

        records = records[-limit:]

        return [
            {
                "timestamp": r.timestamp,
                "latency_ms": r.latency_ms,
                "confidence": r.top_confidence,
                "success": r.success,
            }
            for r in records
        ]
