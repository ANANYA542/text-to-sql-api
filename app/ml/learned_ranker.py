"""
Learned Table Ranker using Classical ML (LightGBM, XGBoost, Random Forest).

This module trains a ranking model that replaces the heuristic reranking rules
in the retrieval engine. It takes the 28 features extracted by FeatureExtractor
and predicts whether each candidate table is relevant to the question.

Design Decisions:
    1. Binary classification (relevant/not) rather than pointwise regression:
       - Simpler to train and evaluate
       - Works well with small datasets (~1,250 samples)
       - Predicted probabilities can be used for ranking

    2. LightGBM as primary model:
       - State-of-the-art for tabular data with small-to-medium datasets
       - Handles feature interactions natively (no need for manual crossing)
       - Faster training than XGBoost for comparable quality
       - Built-in feature importance analysis

    3. Grouped cross-validation:
       - All candidates from the same question stay in the same fold
       - Prevents information leakage across train/val splits
       - Essential for honest evaluation of ranking quality

    4. joblib for serialization:
       - Standard sklearn convention
       - Includes both the model and the feature pipeline
       - Backward-compatible across minor version bumps

Evaluation Metrics:
    - Recall@K: Fraction of gold tables appearing in top-K predictions
    - NDCG@K: Normalized Discounted Cumulative Gain (standard ranking metric)
    - Precision/Recall/F1: Per-sample classification metrics
    - AUC-ROC: Discrimination ability across thresholds
"""

from __future__ import annotations

import logging
import time
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import joblib

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)
from sklearn.preprocessing import StandardScaler

from app.ml.feature_engineering import FEATURE_NAMES, NUM_FEATURES
from app.ml.experiment_tracker import ExperimentTracker, ExperimentRun

logger = logging.getLogger(__name__)

# Default model artifact paths
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "ranker_v1.joblib"

# Suppress convergence warnings during hyperparameter search
warnings.filterwarnings("ignore", category=UserWarning)


def _compute_recall_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
) -> float:
    """
    Computes Recall@K for a ranking task, grouped by query.

    For each unique group (question):
        1. Sort candidates by predicted score (descending)
        2. Take top-K candidates
        3. Recall@K = |{gold tables in top-K}| / |{all gold tables}|
    Average across all groups.

    This is the primary metric for our task because it directly measures
    whether the correct tables end up in the top-5 retrieved results.
    """
    unique_groups = np.unique(groups)
    recalls = []

    for group_id in unique_groups:
        mask = groups == group_id
        group_true = y_true[mask]
        group_scores = y_scores[mask]

        # Skip groups with no positive labels (shouldn't happen in our data)
        if group_true.sum() == 0:
            continue

        # Sort by predicted score (descending) and take top-K
        sorted_indices = np.argsort(-group_scores)
        top_k_labels = group_true[sorted_indices[:k]]

        # Recall@K = true positives in top-K / total positives
        recall = top_k_labels.sum() / group_true.sum()
        recalls.append(recall)

    return float(np.mean(recalls)) if recalls else 0.0


def _compute_ndcg_at_k(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
) -> float:
    """
    Computes NDCG@K (Normalized Discounted Cumulative Gain) for ranking.

    NDCG measures not just whether relevant items appear in top-K,
    but whether they appear at higher positions (positions 1-2 are worth
    more than positions 4-5).

    Formula:
        DCG@K = Σ (2^rel_i - 1) / log2(i + 1) for i in 1..K
        NDCG@K = DCG@K / ideal_DCG@K
    """
    unique_groups = np.unique(groups)
    ndcgs = []

    for group_id in unique_groups:
        mask = groups == group_id
        group_true = y_true[mask]
        group_scores = y_scores[mask]

        if group_true.sum() == 0:
            continue

        # Actual DCG with predicted ordering
        sorted_indices = np.argsort(-group_scores)
        sorted_labels = group_true[sorted_indices[:k]]
        dcg = sum(
            (2 ** rel - 1) / np.log2(pos + 2)
            for pos, rel in enumerate(sorted_labels)
        )

        # Ideal DCG (perfect ordering)
        ideal_sorted = np.sort(group_true)[::-1][:k]
        idcg = sum(
            (2 ** rel - 1) / np.log2(pos + 2)
            for pos, rel in enumerate(ideal_sorted)
        )

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def _evaluate_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
    prefix: str = "val",
) -> dict[str, float]:
    """
    Computes all evaluation metrics for a set of predictions.

    Returns a dict with keys like "val_recall_at_5", "val_ndcg_at_5", etc.
    """
    y_pred = (y_scores >= 0.5).astype(int)

    metrics = {
        f"{prefix}_recall_at_5": _compute_recall_at_k(y_true, y_scores, groups, k=5),
        f"{prefix}_recall_at_10": _compute_recall_at_k(y_true, y_scores, groups, k=10),
        f"{prefix}_ndcg_at_5": _compute_ndcg_at_k(y_true, y_scores, groups, k=5),
        f"{prefix}_ndcg_at_10": _compute_ndcg_at_k(y_true, y_scores, groups, k=10),
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_precision": float(precision_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_recall": float(recall_score(y_true, y_pred, zero_division=0)),
        f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    # AUC-ROC (needs both classes present)
    if len(np.unique(y_true)) > 1:
        metrics[f"{prefix}_auc_roc"] = float(roc_auc_score(y_true, y_scores))
    else:
        metrics[f"{prefix}_auc_roc"] = 0.0

    return metrics


class LearnedRanker:
    """
    Classical ML ranker for table relevance prediction.

    Supports LightGBM, XGBoost, Gradient Boosting (sklearn), and Random Forest.
    Wraps training, cross-validation, prediction, feature importance analysis,
    and model persistence.

    Usage (Training):
        ranker = LearnedRanker(model_type="lightgbm")
        ranker.train(X_train, y_train, groups_train, cv_folds=5)
        ranker.save("models/ranker_v1.joblib")

    Usage (Inference):
        ranker = LearnedRanker.load("models/ranker_v1.joblib")
        scores = ranker.predict(X_candidates)
        top_k_indices = ranker.predict_top_k(X_candidates, k=5)
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        hyperparameters: dict[str, Any] | None = None,
    ):
        """
        Args:
            model_type: One of "lightgbm", "xgboost", "gradient_boosting", "random_forest".
            hyperparameters: Model-specific hyperparameters. If None, uses tuned defaults.
        """
        self.model_type = model_type
        self.hyperparameters = hyperparameters or {}
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importances_: dict[str, float] = {}
        self.training_metrics: dict[str, float] = {}
        self.cv_fold_metrics: list[dict[str, float]] = []

    def _create_model(self) -> Any:
        """
        Creates the underlying sklearn-compatible model.

        Default hyperparameters are tuned for our specific problem:
        - Small dataset (~1,250 samples)
        - 28 features
        - Binary classification with ~20% positive rate
        """
        if self.model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                defaults = {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "min_child_samples": 10,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "verbose": -1,
                    "n_jobs": -1,
                }
                defaults.update(self.hyperparameters)
                return LGBMClassifier(**defaults)
            except Exception as e:
                logger.warning(f"LightGBM could not be imported/loaded: {e}. Falling back to GradientBoosting.")
                self.model_type = "gradient_boosting"
                return self._create_model()

        elif self.model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                defaults = {
                    "n_estimators": 200,
                    "max_depth": 5,
                    "learning_rate": 0.05,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_alpha": 0.1,
                    "reg_lambda": 1.0,
                    "random_state": 42,
                    "eval_metric": "logloss",
                    "verbosity": 0,
                    "n_jobs": -1,
                }
                defaults.update(self.hyperparameters)
                return XGBClassifier(**defaults)
            except Exception as e:
                logger.warning(f"XGBoost could not be imported/loaded: {e}. Falling back to GradientBoosting.")
                self.model_type = "gradient_boosting"
                return self._create_model()

        elif self.model_type == "random_forest":
            defaults = {
                "n_estimators": 200,
                "max_depth": 8,
                "min_samples_leaf": 5,
                "max_features": "sqrt",
                "random_state": 42,
                "n_jobs": -1,
            }
            defaults.update(self.hyperparameters)
            return RandomForestClassifier(**defaults)

        else:  # gradient_boosting (sklearn default, always available)
            defaults = {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "min_samples_leaf": 10,
                "random_state": 42,
            }
            defaults.update(self.hyperparameters)
            return GradientBoostingClassifier(**defaults)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        cv_folds: int = 5,
        tracker: ExperimentTracker | None = None,
    ) -> dict[str, float]:
        """
        Trains the model with grouped cross-validation.

        Args:
            X: Feature matrix of shape (num_samples, NUM_FEATURES).
            y: Binary labels of shape (num_samples,). 1 = relevant, 0 = not.
            groups: Group IDs of shape (num_samples,). Samples from the same
                question share the same group ID.
            cv_folds: Number of cross-validation folds.
            tracker: Optional experiment tracker for logging the run.

        Returns:
            Dictionary of average validation metrics across folds.
        """
        start_time = time.time()
        logger.info(f"Training {self.model_type} ranker: {X.shape[0]} samples, "
                     f"{X.shape[1]} features, {cv_folds}-fold CV")

        # Validate inputs
        assert X.shape[1] == NUM_FEATURES, (
            f"Expected {NUM_FEATURES} features, got {X.shape[1]}"
        )
        assert len(y) == len(X), "X and y must have the same number of samples"
        assert len(groups) == len(X), "groups must have the same length as X"

        # Scale features (important for gradient boosting convergence)
        X_scaled = self.scaler.fit_transform(X)

        # Grouped K-Fold cross-validation
        # Using GroupKFold ensures no question appears in both train and val
        unique_groups = np.unique(groups)
        actual_folds = min(cv_folds, len(unique_groups))

        if actual_folds < cv_folds:
            logger.warning(
                f"Only {len(unique_groups)} unique groups available. "
                f"Reducing CV folds from {cv_folds} to {actual_folds}."
            )

        gkf = GroupKFold(n_splits=actual_folds)

        fold_metrics = []
        best_fold_model = None
        best_fold_score = -1.0

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_scaled, y, groups)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            groups_val = groups[val_idx]

            # Create a fresh model for each fold
            model = self._create_model()
            model.fit(X_train, y_train)

            # Predict probabilities for ranking
            if hasattr(model, "predict_proba"):
                y_scores = model.predict_proba(X_val)[:, 1]
            else:
                y_scores = model.decision_function(X_val)

            # Compute metrics for this fold
            metrics = _evaluate_metrics(y_val, y_scores, groups_val, prefix="val")
            fold_metrics.append(metrics)

            recall_5 = metrics["val_recall_at_5"]
            logger.info(
                f"  Fold {fold_idx + 1}/{actual_folds}: "
                f"Recall@5={recall_5:.4f}, "
                f"NDCG@5={metrics['val_ndcg_at_5']:.4f}, "
                f"AUC={metrics['val_auc_roc']:.4f}"
            )

            # Track best fold model
            if recall_5 > best_fold_score:
                best_fold_score = recall_5
                best_fold_model = model

        # Average metrics across folds
        avg_metrics = {}
        for key in fold_metrics[0]:
            values = [fm[key] for fm in fold_metrics]
            avg_metrics[key] = float(np.mean(values))
            avg_metrics[f"{key}_std"] = float(np.std(values))

        self.cv_fold_metrics = fold_metrics
        self.training_metrics = avg_metrics

        # Final model: retrain on all data with the best fold's hyperparameters
        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        # Extract feature importances
        self._extract_feature_importances()

        training_duration = time.time() - start_time

        logger.info(
            f"Training complete in {training_duration:.1f}s. "
            f"Avg Recall@5={avg_metrics['val_recall_at_5']:.4f} "
            f"(±{avg_metrics.get('val_recall_at_5_std', 0):.4f})"
        )

        # Log to experiment tracker if provided
        if tracker:
            run = ExperimentRun(
                model_type=self.model_type,
                description=f"{self.model_type} ranker with {NUM_FEATURES} features",
                num_training_samples=len(y),
                num_features=NUM_FEATURES,
                cv_folds=actual_folds,
                hyperparameters=self.hyperparameters or self._get_model_params(),
                val_metrics=avg_metrics,
                cv_fold_metrics=fold_metrics,
                feature_importances=self.feature_importances_,
                training_duration_seconds=training_duration,
            )
            tracker.log_run(run)

        return avg_metrics

    def _extract_feature_importances(self):
        """Extracts and normalizes feature importances from the trained model."""
        if self.model is None:
            return

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            # Normalize to sum to 1.0
            total = sum(importances)
            if total > 0:
                importances = importances / total

            self.feature_importances_ = {
                name: round(float(imp), 6)
                for name, imp in sorted(
                    zip(FEATURE_NAMES, importances),
                    key=lambda x: x[1],
                    reverse=True,
                )
            }
        else:
            self.feature_importances_ = {}

    def _get_model_params(self) -> dict[str, Any]:
        """Gets the actual model parameters (useful for logging defaults)."""
        if self.model and hasattr(self.model, "get_params"):
            return {k: v for k, v in self.model.get_params().items()
                    if not callable(v)}
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts relevance scores for a batch of candidates.

        Args:
            X: Feature matrix of shape (num_candidates, NUM_FEATURES).

        Returns:
            Array of relevance scores (probabilities) of shape (num_candidates,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call train() or load() first.")

        X_scaled = self.scaler.transform(X)

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.decision_function(X_scaled)

    def predict_top_k(
        self,
        X: np.ndarray,
        candidate_names: list[str],
        k: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Predicts and returns the top-K candidates with their scores.

        Args:
            X: Feature matrix of shape (num_candidates, NUM_FEATURES).
            candidate_names: Names of the candidate tables (same order as X).
            k: Number of top candidates to return.

        Returns:
            List of (table_name, score) tuples, sorted by score descending.
        """
        scores = self.predict(X)
        ranked_indices = np.argsort(-scores)[:k]

        return [
            (candidate_names[idx], float(scores[idx]))
            for idx in ranked_indices
        ]

    def save(self, path: str | Path | None = None):
        """
        Saves the trained model, scaler, and metadata to disk.

        The saved artifact includes:
        - The trained model (LightGBM/XGBoost/etc.)
        - The fitted StandardScaler
        - Feature names and importances
        - Model type identifier
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        path = Path(path) if path else DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "model_type": self.model_type,
            "feature_names": FEATURE_NAMES,
            "feature_importances": self.feature_importances_,
            "training_metrics": self.training_metrics,
        }

        joblib.dump(artifact, path)
        logger.info(f"Saved {self.model_type} ranker to {path}")

    @classmethod
    def load(cls, path: str | Path | None = None) -> "LearnedRanker":
        """
        Loads a trained model from disk.

        Returns a LearnedRanker instance ready for prediction.
        """
        path = Path(path) if path else DEFAULT_MODEL_PATH

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        artifact = joblib.load(path)

        ranker = cls(model_type=artifact["model_type"])
        ranker.model = artifact["model"]
        ranker.scaler = artifact["scaler"]
        ranker.feature_importances_ = artifact.get("feature_importances", {})
        ranker.training_metrics = artifact.get("training_metrics", {})
        ranker.is_fitted = True

        logger.info(f"Loaded {ranker.model_type} ranker from {path}")
        return ranker

    def get_feature_importance_report(self, top_n: int = 10) -> str:
        """
        Returns a human-readable feature importance report.

        Useful for debugging and understanding which signals the model
        relies on most heavily.
        """
        if not self.feature_importances_:
            return "No feature importances available (model not trained)."

        lines = [f"\nTop-{top_n} Feature Importances ({self.model_type}):"]
        lines.append("-" * 50)

        for i, (name, importance) in enumerate(self.feature_importances_.items()):
            if i >= top_n:
                break
            bar = "█" * int(importance * 100)
            lines.append(f"  {name:40s} {importance:.4f} {bar}")

        return "\n".join(lines)
