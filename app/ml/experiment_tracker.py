"""
Lightweight Experiment Tracker for ML Training Runs.

Tracks model training experiments without requiring external services (no MLflow,
no W&B). All experiment data is stored as JSON files in .cache/experiments/.

Design Decisions:
    - JSON over SQLite: Experiments are few (dozens, not millions). JSON is
      human-readable, git-diffable, and doesn't need a schema migration strategy.
    - File-per-experiment: Each run gets its own file for easy browsing, rather
      than appending to a single growing file (which has corruption risk).
    - No global state: The tracker is a regular class instance, not a singleton.
      This makes testing straightforward and avoids import-time side effects.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

# Default directory for experiment logs.
DEFAULT_EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / ".cache" / "experiments"


@dataclass
class ExperimentRun:
    """
    A single experiment run record.

    This captures everything needed to reproduce and compare runs:
    - What model was used
    - What hyperparameters were set
    - What metrics were achieved
    - Which features were most important
    - When was it run
    """
    # Identification
    run_id: str = ""
    timestamp: str = ""
    model_type: str = ""  # e.g., "lightgbm", "xgboost", "random_forest"
    description: str = ""

    # Data configuration
    dataset: str = "beaver_dw"
    num_training_samples: int = 0
    num_features: int = 0
    cv_folds: int = 5
    train_test_split_ratio: float = 0.8

    # Hyperparameters (stored as a flexible dict to support any model type)
    hyperparameters: dict[str, Any] = field(default_factory=dict)

    # Metrics (train/val/test)
    train_metrics: dict[str, float] = field(default_factory=dict)
    val_metrics: dict[str, float] = field(default_factory=dict)
    test_metrics: dict[str, float] = field(default_factory=dict)

    # Per-fold cross-validation metrics
    cv_fold_metrics: list[dict[str, float]] = field(default_factory=list)

    # Feature importances (ordered by importance)
    feature_importances: dict[str, float] = field(default_factory=dict)

    # Model artifact path
    model_path: str = ""

    # Training duration
    training_duration_seconds: float = 0.0

    # Notes / observations
    notes: str = ""


class ExperimentTracker:
    """
    Tracks ML experiment runs to a local JSON file store.

    Usage:
        tracker = ExperimentTracker()
        run = ExperimentRun(
            model_type="lightgbm",
            hyperparameters={"n_estimators": 100, "max_depth": 5},
            ...
        )
        tracker.log_run(run)

        # Later, retrieve and compare:
        all_runs = tracker.list_runs()
        best_run = tracker.get_best_run(metric="val_recall_at_5")
    """

    def __init__(self, experiment_dir: Path | str | None = None):
        """
        Args:
            experiment_dir: Directory to store experiment JSON files.
                Defaults to .cache/experiments/ relative to the project root.
        """
        self.experiment_dir = Path(experiment_dir) if experiment_dir else DEFAULT_EXPERIMENT_DIR
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ExperimentTracker initialized. Storage: {self.experiment_dir}")

    def _generate_run_id(self, model_type: str) -> str:
        """Generates a unique, sortable run ID.

        Format: {model_type}_{YYYYMMDD_HHMMSS}
        Using timestamps ensures chronological sorting and human-readability.
        """
        now = datetime.now(timezone.utc)
        timestamp_str = now.strftime("%Y%m%d_%H%M%S")
        return f"{model_type}_{timestamp_str}"

    def log_run(self, run: ExperimentRun) -> str:
        """
        Saves an experiment run to disk.

        Automatically fills in run_id and timestamp if not provided.

        Returns:
            The run_id of the saved experiment.
        """
        # Auto-fill identification fields
        if not run.run_id:
            run.run_id = self._generate_run_id(run.model_type)
        if not run.timestamp:
            run.timestamp = datetime.now(timezone.utc).isoformat()

        # Serialize to JSON
        run_path = self.experiment_dir / f"{run.run_id}.json"
        run_dict = asdict(run)

        try:
            with open(run_path, "w") as f:
                json.dump(run_dict, f, indent=2, default=str)
            logger.info(f"Logged experiment run: {run.run_id} → {run_path}")
        except Exception as e:
            logger.error(f"Failed to save experiment run {run.run_id}: {e}")
            raise

        return run.run_id

    def load_run(self, run_id: str) -> ExperimentRun | None:
        """Loads a single experiment run by its ID."""
        run_path = self.experiment_dir / f"{run_id}.json"
        if not run_path.exists():
            logger.warning(f"Experiment run not found: {run_id}")
            return None

        try:
            with open(run_path, "r") as f:
                data = json.load(f)
            return ExperimentRun(**data)
        except Exception as e:
            logger.error(f"Failed to load experiment run {run_id}: {e}")
            return None

    def list_runs(self, model_type: str | None = None) -> list[ExperimentRun]:
        """
        Lists all experiment runs, optionally filtered by model type.

        Returns runs sorted by timestamp (newest first).
        """
        runs = []
        for file_path in sorted(self.experiment_dir.glob("*.json"), reverse=True):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                run = ExperimentRun(**data)
                if model_type is None or run.model_type == model_type:
                    runs.append(run)
            except Exception as e:
                logger.warning(f"Skipping corrupt experiment file {file_path}: {e}")

        return runs

    def get_best_run(
        self,
        metric: str = "val_recall_at_5",
        model_type: str | None = None,
        higher_is_better: bool = True,
    ) -> ExperimentRun | None:
        """
        Returns the experiment run with the best value for a given metric.

        The metric is looked up in val_metrics first, then test_metrics, then train_metrics.
        """
        runs = self.list_runs(model_type=model_type)
        if not runs:
            return None

        def get_metric_value(run: ExperimentRun) -> float:
            for metrics_dict in [run.val_metrics, run.test_metrics, run.train_metrics]:
                if metric in metrics_dict:
                    return metrics_dict[metric]
            return float("-inf") if higher_is_better else float("inf")

        return max(runs, key=get_metric_value) if higher_is_better else min(runs, key=get_metric_value)

    def get_comparison_table(self, metric_keys: list[str] | None = None) -> list[dict[str, Any]]:
        """
        Returns a list of dicts suitable for rendering as a comparison table.

        Each dict contains: run_id, model_type, timestamp, and selected metrics.
        Used by the web UI dashboard and README generation.
        """
        if metric_keys is None:
            metric_keys = [
                "val_recall_at_5", "val_recall_at_10", "val_ndcg_at_5",
                "val_accuracy", "val_f1"
            ]

        runs = self.list_runs()
        table = []
        for run in runs:
            row: dict[str, Any] = {
                "run_id": run.run_id,
                "model_type": run.model_type,
                "timestamp": run.timestamp,
                "training_duration_s": round(run.training_duration_seconds, 1),
                "num_samples": run.num_training_samples,
            }
            # Merge val and test metrics
            all_metrics = {**run.val_metrics, **run.test_metrics}
            for key in metric_keys:
                row[key] = round(all_metrics.get(key, 0.0), 4)
            table.append(row)

        return table

    def delete_run(self, run_id: str) -> bool:
        """Deletes an experiment run file."""
        run_path = self.experiment_dir / f"{run_id}.json"
        if run_path.exists():
            run_path.unlink()
            logger.info(f"Deleted experiment run: {run_id}")
            return True
        return False
