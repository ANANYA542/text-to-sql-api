"""
Standalone Training Script for the Learned Table Ranker.

This script automates the full ML training pipeline:
    1. Load the Beaver dataset (queries + schema)
    2. Generate training data from gold SQL annotations
    3. Extract features for all (question, candidate_table) pairs
    4. Train and cross-validate multiple model types
    5. Compare results and save the best model
    6. Log everything to the experiment tracker

Usage:
    python -m app.ml.train_ranker
    python -m app.ml.train_ranker --model lightgbm --cv-folds 5
    python -m app.ml.train_ranker --model all --compare

The script is designed to run offline (no API calls during training).
All retrieval pipeline scores are precomputed and cached.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Configure logging before importing app modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Project root for imports
import os
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")


def generate_training_data(
    max_queries: int = 50,
    candidates_per_query: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Generates training data for the learned ranker.

    Process:
        1. Load Beaver queries with gold SQL
        2. Parse gold SQL to extract correct (gold) tables
        3. Run the retrieval pipeline to get candidate tables
        4. For each candidate, extract features and assign labels
        5. Return (X, y, groups, questions) ready for training

    Returns:
        X: Feature matrix of shape (num_samples, 28)
        y: Binary labels of shape (num_samples,)
        groups: Question group IDs of shape (num_samples,)
        questions: List of question texts (one per group)
    """
    from app.retrieval.schema_loader import load_beaver_schema, load_beaver_queries
    from app.retrieval.engine import RetrievalEngine
    from app.ml.feature_engineering import (
        FeatureExtractor, PipelineScores, FEATURE_NAMES, NUM_FEATURES,
    )
    from scripts.test_retrieval_accuracy import extract_tables_and_ctes

    logger.info("=== Generating Training Data ===")

    # Load schema and initialize engine
    schema = load_beaver_schema(split="dw")
    valid_schema_tables = {name.upper() for name in schema.keys()}
    engine = RetrievalEngine(schema)

    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        schema=engine.schema,
        relations=engine.relations,
        table_names=engine.table_names,
    )

    # Load queries
    query_ds = load_beaver_queries(split="dw")

    eval_queries = []
    for row in query_ds:
        if len(eval_queries) >= max_queries:
            break
        if row.get("question") and row.get("sql"):
            eval_queries.append(row)

    logger.info(f"Loaded {len(eval_queries)} queries for training data generation.")

    # Generate training samples
    all_features = []
    all_labels = []
    all_groups = []
    question_texts = []

    for q_idx, row in enumerate(eval_queries):
        question = row["question"]
        gold_sql = row["sql"]

        # Parse gold tables from SQL
        raw_tables, ctes = extract_tables_and_ctes(gold_sql)
        tables_no_ctes = raw_tables - ctes
        gold_tables = {t for t in tables_no_ctes if t in valid_schema_tables}

        if not gold_tables:
            logger.debug(f"Skipping Q#{q_idx}: no gold tables after filtering.")
            continue

        question_texts.append(question)

        # Run the retrieval pipeline to get candidates
        # We use the engine's internal methods to capture intermediate scores
        expanded_query = engine.expand_query(question)

        # BM25 scores
        bm25_raw_scores = engine.bm25.get_scores(expanded_query.lower().split())
        bm25_scores_dict = {}
        bm25_ranks_dict = {}
        bm25_pairs = sorted(
            zip(engine.table_names, bm25_raw_scores),
            key=lambda x: x[1], reverse=True,
        )
        for rank, (tname, score) in enumerate(bm25_pairs):
            bm25_scores_dict[tname] = float(score)
            bm25_ranks_dict[tname] = rank + 1

        # Normalize BM25 scores to [0, 1]
        bm25_max = max(bm25_raw_scores) if len(bm25_raw_scores) > 0 and max(bm25_raw_scores) > 0 else 1.0
        bm25_min = min(bm25_raw_scores) if len(bm25_raw_scores) > 0 else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        for tname in engine.table_names:
            raw = bm25_scores_dict.get(tname, 0.0)
            bm25_scores_dict[tname] = (raw - bm25_min) / bm25_range

        # Cosine similarity scores
        from sentence_transformers import util as st_util
        q_embedding = engine.bi_encoder.encode(question, convert_to_tensor=True)
        cosine_raw = st_util.cos_sim(q_embedding, engine.table_embeddings)[0].tolist()
        cosine_scores_dict = {}
        cosine_ranks_dict = {}
        cosine_pairs = sorted(
            zip(engine.table_names, cosine_raw),
            key=lambda x: x[1], reverse=True,
        )
        for rank, (tname, score) in enumerate(cosine_pairs):
            cosine_scores_dict[tname] = float(score)
            cosine_ranks_dict[tname] = rank + 1

        # Get top candidates from hybrid search
        candidates = engine.get_hybrid_candidates(question, top_k=candidates_per_query)
        candidate_names = [name for name, _ in candidates]

        # Cross-encoder scores for candidates
        pairs_for_ce = []
        for name in candidate_names:
            desc = engine.schema.get(name, name)
            clean_desc = engine.clean_descriptions.get(name, desc)
            pairs_for_ce.append([question, clean_desc])

        ce_logits_raw = np.asarray(engine.cross_encoder.predict(pairs_for_ce), dtype=float)
        ce_logits_raw = np.nan_to_num(ce_logits_raw, nan=0.0, posinf=0.0, neginf=0.0)
        ce_logits_dict = {}
        for i, name in enumerate(candidate_names):
            ce_logits_dict[name] = float(ce_logits_raw[i])

        # Build PipelineScores object
        pipeline_scores = PipelineScores(
            bm25_scores=bm25_scores_dict,
            cosine_scores=cosine_scores_dict,
            cross_encoder_logits=ce_logits_dict,
            bm25_ranks=bm25_ranks_dict,
            cosine_ranks=cosine_ranks_dict,
            llm_expansion_text=expanded_query,
        )

        # Extract features for each candidate
        for table_name in candidate_names:
            feat_dict = feature_extractor.extract(
                question, table_name, pipeline_scores, candidate_names,
            )

            # Convert to feature vector in canonical order
            feat_vector = [feat_dict.get(fname, 0.0) for fname in FEATURE_NAMES]
            all_features.append(feat_vector)

            # Label: 1 if this table is in the gold set, 0 otherwise
            label = 1.0 if table_name.upper() in gold_tables else 0.0
            all_labels.append(label)

            # Group ID = question index
            all_groups.append(q_idx)

        if (q_idx + 1) % 10 == 0:
            logger.info(f"  Processed {q_idx + 1}/{len(eval_queries)} queries...")

    X = np.array(all_features, dtype=np.float64)
    y = np.array(all_labels, dtype=np.float64)
    groups = np.array(all_groups, dtype=np.int64)

    # Report statistics
    num_positive = int(y.sum())
    num_negative = len(y) - num_positive
    logger.info(
        f"Training data generated: {len(y)} samples "
        f"({num_positive} positive, {num_negative} negative, "
        f"ratio={num_positive / len(y):.2%}), "
        f"{len(question_texts)} unique questions."
    )

    return X, y, groups, question_texts


def train_single_model(
    model_type: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_folds: int = 5,
    save_path: str | None = None,
) -> dict[str, float]:
    """
    Trains a single model type and optionally saves it.

    Returns the average validation metrics.
    """
    from app.ml.learned_ranker import LearnedRanker
    from app.ml.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker()
    ranker = LearnedRanker(model_type=model_type)

    metrics = ranker.train(X, y, groups, cv_folds=cv_folds, tracker=tracker)

    # Print feature importance report
    report = ranker.get_feature_importance_report(top_n=15)
    print(report)

    if save_path:
        ranker.save(save_path)

    return metrics


def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv_folds: int = 5,
) -> dict[str, dict[str, float]]:
    """
    Trains all supported model types and compares them.

    Returns a dict mapping model_type -> metrics for comparison.
    """
    model_types = ["gradient_boosting", "random_forest"]

    # Try to import optional packages
    try:
        import lightgbm
        model_types.insert(0, "lightgbm")
    except Exception as e:
        logger.warning(f"LightGBM could not be imported/loaded: {e}. Skipping.")

    try:
        import xgboost
        model_types.insert(1, "xgboost")
    except Exception as e:
        logger.warning(f"XGBoost could not be imported/loaded: {e}. Skipping.")

    results = {}
    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"Training: {model_type.upper()}")
        print(f"{'=' * 60}")

        metrics = train_single_model(
            model_type=model_type,
            X=X, y=y, groups=groups,
            cv_folds=cv_folds,
        )
        results[model_type] = metrics

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("MODEL COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Model':<25} {'Recall@5':>10} {'Recall@10':>10} {'NDCG@5':>10} {'AUC':>10} {'F1':>10}")
    print("-" * 80)

    best_model = None
    best_recall = -1.0

    for model_type, metrics in results.items():
        r5 = metrics.get("val_recall_at_5", 0.0)
        r10 = metrics.get("val_recall_at_10", 0.0)
        ndcg5 = metrics.get("val_ndcg_at_5", 0.0)
        auc = metrics.get("val_auc_roc", 0.0)
        f1 = metrics.get("val_f1", 0.0)
        print(f"{model_type:<25} {r5:>10.4f} {r10:>10.4f} {ndcg5:>10.4f} {auc:>10.4f} {f1:>10.4f}")

        if r5 > best_recall:
            best_recall = r5
            best_model = model_type

    print(f"\n✓ Best model: {best_model} (Recall@5 = {best_recall:.4f})")

    # Save the best model
    if best_model:
        print(f"\nSaving best model ({best_model})...")
        from app.ml.learned_ranker import LearnedRanker, DEFAULT_MODEL_PATH
        ranker = LearnedRanker(model_type=best_model)
        ranker.train(X, y, groups, cv_folds=cv_folds)
        ranker.save(DEFAULT_MODEL_PATH)
        print(f"Saved to: {DEFAULT_MODEL_PATH}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train the Learned Table Ranker for Text-to-SQL retrieval."
    )
    parser.add_argument(
        "--model", type=str, default="gradient_boosting",
        choices=["lightgbm", "xgboost", "gradient_boosting", "random_forest", "all"],
        help="Model type to train. Use 'all' to train and compare all types.",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of cross-validation folds (default: 5).",
    )
    parser.add_argument(
        "--max-queries", type=int, default=50,
        help="Maximum number of queries to use for training data (default: 50).",
    )
    parser.add_argument(
        "--candidates", type=int, default=25,
        help="Number of candidate tables per query (default: 25).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for the trained model (default: models/ranker_v1.joblib).",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Text-to-SQL Learned Table Ranker — Training Pipeline")
    print("=" * 60)

    # Step 1: Generate training data
    X, y, groups, questions = generate_training_data(
        max_queries=args.max_queries,
        candidates_per_query=args.candidates,
    )

    # Step 2: Train model(s)
    if args.model == "all":
        train_all_models(X, y, groups, cv_folds=args.cv_folds)
    else:
        from app.ml.learned_ranker import DEFAULT_MODEL_PATH
        save_path = args.output or str(DEFAULT_MODEL_PATH)
        train_single_model(
            model_type=args.model,
            X=X, y=y, groups=groups,
            cv_folds=args.cv_folds,
            save_path=save_path,
        )

    print("\n✓ Training pipeline complete.")


if __name__ == "__main__":
    main()
