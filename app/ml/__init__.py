# Machine Learning pipeline for learned table reranking.
# This package contains:
#   - feature_engineering: Extract 28+ features for (question, table) pairs
#   - learned_ranker: Train and predict with LightGBM/XGBoost ranking models
#   - experiment_tracker: Log training runs, metrics, and feature importances
#   - train_ranker: Standalone training + evaluation script
