"""
Feature Engineering for Learned Table Reranking.

This module extracts 28 handcrafted features for each (question, candidate_table)
pair. These features are used by the LearnedRanker (LightGBM/XGBoost) to predict
whether a table is relevant to a given natural language question.

Design Principles:
    1. Each feature captures a distinct signal (lexical, semantic, structural,
       query-level, or interaction).
    2. Features are returned as named dictionaries for debuggability — you can
       print features["bm25_score"] and understand what's happening.
    3. No feature leaks future information (e.g., no access to gold tables).
    4. All features are numeric (float) for direct consumption by tree models.

Feature Categories:
    - LEXICAL: Exact text overlap between question and table description
    - SEMANTIC: Embedding-based similarity scores
    - STRUCTURAL: Schema topology (columns, FKs, graph connectivity)
    - QUERY: Characteristics of the question itself
    - INTERACTION: Cross-signal combinations and rank-based features
"""

from __future__ import annotations

import math
import re
import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# SQL aggregation keywords that signal specific query types.
# These help the model learn that "how many" questions need COUNT tables.
AGGREGATION_KEYWORDS = frozenset({
    "count", "average", "avg", "sum", "total", "maximum", "max",
    "minimum", "min", "mean", "stddev", "variance", "median",
})

JOIN_KEYWORDS = frozenset({
    "join", "combine", "merge", "link", "relate", "connect",
    "associated", "corresponding", "along with", "together with",
})

SUBQUERY_KEYWORDS = frozenset({
    "for each", "per", "among", "within", "those that",
    "having", "where the", "only those", "excluding",
})

# Feature names in a fixed order for consistent column mapping.
# This ordering is used when converting feature dicts to numpy arrays.
FEATURE_NAMES: list[str] = [
    # Lexical features (7)
    "bm25_score",
    "tfidf_cosine",
    "jaccard_word",
    "jaccard_char_3gram",
    "exact_table_name_match",
    "partial_table_name_match",
    "query_table_token_overlap",
    # Semantic features (3)
    "biencoder_cosine",
    "cross_encoder_logit",
    "cross_encoder_calibrated",
    # Structural features (5)
    "num_columns",
    "num_fk_relations",
    "schema_graph_degree",
    "has_join_key_overlap_with_candidates",
    "num_shared_join_keys_with_candidates",
    # Query features (6)
    "query_length_words",
    "query_length_chars",
    "has_aggregation_keyword",
    "has_join_keyword",
    "has_subquery_indicator",
    "num_sql_keywords_detected",
    # Interaction features (7)
    "bm25_x_cosine",
    "bm25_rank",
    "cosine_rank",
    "rank_difference",
    "score_ratio",
    "llm_expansion_overlap",
    "category_prefix_match",
]

NUM_FEATURES = len(FEATURE_NAMES)


@dataclass
class PipelineScores:
    """
    Intermediate scores from the retrieval pipeline stages.

    These are passed into the feature extractor so it doesn't need to
    re-run BM25/cosine/cross-encoder — those are expensive computations
    that have already been performed by the RetrievalEngine.
    """
    bm25_scores: dict[str, float] = field(default_factory=dict)
    cosine_scores: dict[str, float] = field(default_factory=dict)
    cross_encoder_logits: dict[str, float] = field(default_factory=dict)
    bm25_ranks: dict[str, int] = field(default_factory=dict)
    cosine_ranks: dict[str, int] = field(default_factory=dict)
    llm_expansion_text: str = ""


def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Extracts character n-grams from a string.

    Character n-grams capture partial token matches that word-level features miss.
    For example, "FCLT_BUILDING" and "facilities building" share the 3-gram "bui".
    """
    text = text.lower()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard similarity = |A ∩ B| / |A ∪ B|.

    Returns 0.0 for empty sets (instead of raising ZeroDivisionError).
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _safe_sigmoid(logit: float) -> float:
    """Numerically stable sigmoid function.

    Prevents overflow for large positive/negative logits.
    """
    if logit >= 0:
        z = math.exp(-logit)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(logit)
        return z / (1.0 + z)


def _count_keyword_matches(text: str, keywords: frozenset[str]) -> int:
    """Counts how many keywords from the set appear in the text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


class FeatureExtractor:
    """
    Extracts features for (question, candidate_table) pairs.

    This class is stateful — it holds references to the schema and retrieval
    engine's precomputed indices (embeddings, BM25 index, relationship graph).
    It does NOT hold any trained model state; that belongs to LearnedRanker.

    Usage:
        extractor = FeatureExtractor(schema, relations, table_names)
        features = extractor.extract(question, "SIS_DEPARTMENT", pipeline_scores)
        # features is a dict[str, float] with 28 entries
    """

    def __init__(
        self,
        schema: dict[str, str],
        relations: dict[str, list[str]],
        table_names: list[str],
    ):
        """
        Args:
            schema: Mapping of table_name -> enriched description string.
            relations: Mapping of TABLE_NAME -> [related TABLE_NAMEs] from FK graph.
            table_names: Ordered list of all table names in the database.
        """
        self.schema = schema
        self.relations = relations
        self.table_names = table_names

        # Precompute column counts per table by counting commas in description.
        # This is an approximation but avoids re-parsing the schema.
        self._column_counts: dict[str, int] = {}
        for tname, desc in schema.items():
            # Count columns by splitting the "Schema structure: [...]" section
            if "Schema structure: [" in desc:
                cols_section = desc.split("Schema structure: [")[1].split("]")[0]
                self._column_counts[tname] = cols_section.count(",") + 1
            elif "Columns:" in desc:
                cols_section = desc.split("Columns:")[1].split(".")[0]
                self._column_counts[tname] = cols_section.count(",") + 1
            else:
                self._column_counts[tname] = 0

        # Precompute FK relation counts and graph degree for each table
        self._fk_counts: dict[str, int] = {}
        self._graph_degrees: dict[str, int] = {}
        for tname in table_names:
            tname_upper = tname.upper()
            neighbors = relations.get(tname_upper, [])
            self._fk_counts[tname] = len(neighbors)
            # Graph degree = number of tables that list this table as a neighbor
            # (both in-degree and out-degree)
            self._graph_degrees[tname] = len(neighbors)

        # Count reverse edges too (tables that point TO this table)
        for tname_upper, neighbors in relations.items():
            for neighbor in neighbors:
                # Find the original-case name
                for tn in table_names:
                    if tn.upper() == neighbor:
                        self._graph_degrees[tn] = self._graph_degrees.get(tn, 0) + 1
                        break

        # Precompute category prefixes from table names
        self._table_prefixes: dict[str, str] = {}
        for tname in table_names:
            parts = tname.upper().split("_")
            self._table_prefixes[tname] = parts[0] if parts else ""

        logger.info(f"FeatureExtractor initialized for {len(table_names)} tables, "
                     f"{NUM_FEATURES} features per pair.")

    def extract(
        self,
        question: str,
        table_name: str,
        pipeline_scores: PipelineScores,
        candidate_tables: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Extracts all features for a single (question, table_name) pair.

        Args:
            question: The natural language question.
            table_name: The candidate table to score.
            pipeline_scores: Pre-computed BM25, cosine, and cross-encoder scores.
            candidate_tables: Other candidate tables in the same retrieval batch
                (used for inter-candidate features like shared join keys).

        Returns:
            Dictionary mapping feature name -> float value.
        """
        question_lower = question.lower()
        table_lower = table_name.lower()
        desc = self.schema.get(table_name, table_name)
        desc_lower = desc.lower()

        # Tokenize question and description for overlap features
        question_tokens = set(re.findall(r'\w+', question_lower))
        desc_tokens = set(re.findall(r'\w+', desc_lower))
        table_name_tokens = set(table_lower.replace("_", " ").split())

        features: dict[str, float] = {}

        # ── LEXICAL FEATURES ──────────────────────────────────────────

        # 1. BM25 score (already computed by the pipeline)
        features["bm25_score"] = pipeline_scores.bm25_scores.get(table_name, 0.0)

        # 2. TF-IDF cosine similarity (we use cosine_scores as proxy since
        #    the engine computes both TF-IDF and bi-encoder cosine)
        features["tfidf_cosine"] = pipeline_scores.cosine_scores.get(table_name, 0.0)

        # 3. Jaccard word overlap between question and table description
        features["jaccard_word"] = _jaccard_similarity(question_tokens, desc_tokens)

        # 4. Character 3-gram Jaccard overlap (catches partial string matches)
        q_ngrams = _char_ngrams(question_lower)
        t_ngrams = _char_ngrams(desc_lower)
        features["jaccard_char_3gram"] = _jaccard_similarity(q_ngrams, t_ngrams)

        # 5. Exact table name match (does the question literally contain the table name?)
        clean_table_name = table_lower.replace("_", " ")
        features["exact_table_name_match"] = float(
            table_lower in question_lower or clean_table_name in question_lower
        )

        # 6. Partial table name match (does any segment of the table name match?)
        parts = table_lower.split("_")
        partial_match = any(
            part in question_lower
            for part in parts
            if len(part) > 3  # Skip short segments like "HR", "GP" that cause false positives
        )
        features["partial_table_name_match"] = float(partial_match)

        # 7. Token overlap count between question and table name tokens
        overlap_count = len(question_tokens & table_name_tokens)
        features["query_table_token_overlap"] = float(overlap_count)

        # ── SEMANTIC FEATURES ─────────────────────────────────────────

        # 8. Bi-encoder cosine similarity (already computed by the pipeline)
        features["biencoder_cosine"] = pipeline_scores.cosine_scores.get(table_name, 0.0)

        # 9. Cross-encoder raw logit
        ce_logit = pipeline_scores.cross_encoder_logits.get(table_name, 0.0)
        features["cross_encoder_logit"] = ce_logit

        # 10. Calibrated cross-encoder probability (sigmoid of shifted logit)
        features["cross_encoder_calibrated"] = _safe_sigmoid(ce_logit)

        # ── STRUCTURAL FEATURES ───────────────────────────────────────

        # 11. Number of columns in the table
        features["num_columns"] = float(self._column_counts.get(table_name, 0))

        # 12. Number of foreign key relationships
        features["num_fk_relations"] = float(self._fk_counts.get(table_name, 0))

        # 13. Schema graph degree (how connected is this table?)
        features["schema_graph_degree"] = float(self._graph_degrees.get(table_name, 0))

        # 14-15. Join key overlap with other candidates in the current batch
        has_overlap = 0.0
        num_shared_keys = 0
        if candidate_tables:
            my_neighbors = set(self.relations.get(table_name.upper(), []))
            for other in candidate_tables:
                if other != table_name and other.upper() in my_neighbors:
                    has_overlap = 1.0
                    num_shared_keys += 1
        features["has_join_key_overlap_with_candidates"] = has_overlap
        features["num_shared_join_keys_with_candidates"] = float(num_shared_keys)

        # ── QUERY FEATURES ────────────────────────────────────────────

        # 16-17. Query length (words and characters)
        features["query_length_words"] = float(len(question_tokens))
        features["query_length_chars"] = float(len(question))

        # 18. Has aggregation keywords (COUNT, AVG, SUM, etc.)
        features["has_aggregation_keyword"] = float(
            _count_keyword_matches(question_lower, AGGREGATION_KEYWORDS) > 0
        )

        # 19. Has join keywords (associated, combine, etc.)
        features["has_join_keyword"] = float(
            _count_keyword_matches(question_lower, JOIN_KEYWORDS) > 0
        )

        # 20. Has subquery indicators (for each, per, etc.)
        features["has_subquery_indicator"] = float(
            _count_keyword_matches(question_lower, SUBQUERY_KEYWORDS) > 0
        )

        # 21. Total SQL keyword count detected
        total_sql_kw = (
            _count_keyword_matches(question_lower, AGGREGATION_KEYWORDS) +
            _count_keyword_matches(question_lower, JOIN_KEYWORDS) +
            _count_keyword_matches(question_lower, SUBQUERY_KEYWORDS)
        )
        features["num_sql_keywords_detected"] = float(total_sql_kw)

        # ── INTERACTION FEATURES ──────────────────────────────────────

        # 22. BM25 × cosine product (captures agreement between lexical and semantic)
        bm25_val = features["bm25_score"]
        cosine_val = features["biencoder_cosine"]
        features["bm25_x_cosine"] = bm25_val * cosine_val

        # 23-25. Rank-based features (position in BM25 vs cosine rankings)
        features["bm25_rank"] = float(
            pipeline_scores.bm25_ranks.get(table_name, len(self.table_names))
        )
        features["cosine_rank"] = float(
            pipeline_scores.cosine_ranks.get(table_name, len(self.table_names))
        )
        features["rank_difference"] = abs(features["bm25_rank"] - features["cosine_rank"])

        # 26. Score ratio (BM25 / cosine, clamped to avoid division by zero)
        if cosine_val > 1e-6:
            features["score_ratio"] = bm25_val / cosine_val
        else:
            features["score_ratio"] = bm25_val * 100.0  # Large ratio if cosine is near-zero

        # 27. LLM expansion keyword overlap (how many LLM-expanded terms match this table?)
        expansion_text = pipeline_scores.llm_expansion_text.lower()
        if expansion_text:
            expansion_tokens = set(expansion_text.split())
            table_all_tokens = desc_tokens | table_name_tokens
            features["llm_expansion_overlap"] = float(
                len(expansion_tokens & table_all_tokens)
            )
        else:
            features["llm_expansion_overlap"] = 0.0

        # 28. Category prefix match (does the table's category prefix match the question domain?)
        prefix = self._table_prefixes.get(table_name, "")
        from app.retrieval.schema_loader import ABBREV_MAP
        prefix_match = 0.0
        if prefix in ABBREV_MAP:
            meaning = ABBREV_MAP[prefix].lower()
            meaning_words = set(meaning.replace(",", "").split())
            # Check if any meaning words appear in the question
            if question_tokens & meaning_words:
                prefix_match = 1.0
        features["category_prefix_match"] = prefix_match

        return features

    def extract_batch(
        self,
        question: str,
        candidate_tables: list[str],
        pipeline_scores: PipelineScores,
    ) -> np.ndarray:
        """
        Extracts features for all candidate tables in a batch.

        Returns a numpy array of shape (num_candidates, NUM_FEATURES)
        with features in the order defined by FEATURE_NAMES.
        """
        feature_matrix = np.zeros((len(candidate_tables), NUM_FEATURES), dtype=np.float64)

        for i, table_name in enumerate(candidate_tables):
            feat_dict = self.extract(
                question, table_name, pipeline_scores, candidate_tables
            )
            for j, fname in enumerate(FEATURE_NAMES):
                feature_matrix[i, j] = feat_dict.get(fname, 0.0)

        return feature_matrix

    def extract_batch_as_dicts(
        self,
        question: str,
        candidate_tables: list[str],
        pipeline_scores: PipelineScores,
    ) -> list[dict[str, float]]:
        """
        Extracts features for all candidate tables as a list of dicts.

        Useful for debugging and experiment logging where you want
        named features rather than a raw numpy array.
        """
        return [
            self.extract(question, table_name, pipeline_scores, candidate_tables)
            for table_name in candidate_tables
        ]
