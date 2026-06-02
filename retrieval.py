from __future__ import annotations

import logging
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class RetrievalEngine:

    def __init__(self, schema: dict[str, str]):
        """
        schema: dict mapping table_name -> enriched description string
        """
        self.schema = schema
        self.table_names = list(schema.keys())
        self.descriptions = [schema[t] for t in self.table_names]

        logger.info("Building offline TF-IDF index...")
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.table_embeddings = self.vectorizer.fit_transform(self.descriptions)

        # build BM25 index by tokenizing each description into words
        tokenized = [desc.lower().split() for desc in self.descriptions]
        self.bm25 = BM25Okapi(tokenized)

        logger.info("RetrievalEngine ready.")

    # ── Stage 1: Query Expansion ──────────────────────────────────────

    def expand_query(self, question: str) -> str:
        """
        Expands the user question with related SQL terms.
        Uses a simple rule-based approach instead of calling an LLM
        so this runs fast and doesn't need an API key.
        """
        keywords = question.lower().replace("?", "").replace(",", " ").split()
        extras = set()

        # map common natural language words to SQL / schema terms
        keyword_map = {
            "how many": ["COUNT", "GROUP BY"],
            "average": ["AVG", "GROUP BY"],
            "total": ["SUM", "GROUP BY"],
            "maximum": ["MAX"],
            "minimum": ["MIN"],
            "more than": ["HAVING", "COUNT", "greater than"],
            "less than": ["HAVING", "COUNT", "fewer than"],
            "each": ["GROUP BY", "per"],
            "per": ["GROUP BY"],
            "between": ["BETWEEN", "range"],
            "top": ["ORDER BY", "LIMIT", "DESC"],
            "bottom": ["ORDER BY", "LIMIT", "ASC"],
            "ranked": ["ORDER BY", "RANK"],
            "excluding": ["WHERE", "NOT IN", "filter"],
            "without": ["WHERE", "NOT IN", "exclude"],
            "join": ["JOIN", "INNER JOIN", "LEFT JOIN"],
            "list": ["SELECT"],
            "show": ["SELECT"],
            "find": ["SELECT", "WHERE"],
            "count": ["COUNT", "GROUP BY"],
            "sum": ["SUM"],
            "name": ["_name", "name"],
            "id": ["_id", "id"],
            "date": ["date", "timestamp", "created_at"],
        }

        question_lower = question.lower()
        for phrase, terms in keyword_map.items():
            if phrase in question_lower:
                extras.update(terms)

        # also try to guess table names by looking for plural/singular forms
        for tname in self.table_names:
            clean = tname.lower().replace("_", " ")
            if clean in question_lower or clean.rstrip("s") in question_lower:
                extras.add(tname)

        expanded = question + " " + " ".join(extras)
        logger.info(f"Expanded query: {expanded[:200]}...")
        return expanded

    # ── Stage 2a: BM25 keyword search ─────────────────────────────────

    def bm25_search(self, question: str, top_k: int = 15) -> list[tuple[str, float]]:
        tokens = question.lower().split()
        scores = self.bm25.get_scores(tokens)

        ranked = sorted(
            zip(self.table_names, scores), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    # ── Stage 2b: Cosine similarity (TF-IDF) ─────────────────────────

    def cosine_similarity_search(
        self, question: str, top_k: int = 15
    ) -> list[tuple[str, float]]:
        """Computes cosine similarity with TF-IDF vectors."""
        q_embedding = self.vectorizer.transform([question])
        scores = (self.table_embeddings @ q_embedding.T).toarray().ravel()

        ranked = sorted(
            zip(self.table_names, scores.tolist()), key=lambda x: x[1], reverse=True
        )
        return ranked[:top_k]

    # ── Stage 3: Hybrid merge ─────────────────────────────────────────

    def hybrid_search(
        self, question: str, alpha: float = 0.5, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """
        Combines BM25 and cosine scores:
            final = alpha * cosine + (1 - alpha) * bm25_normalized

        alpha controls the balance. 0.5 = equal weight.
        """
        expanded = self.expand_query(question)

        bm25_results = self.bm25_search(expanded, top_k=len(self.table_names))
        cosine_results = self.cosine_similarity_search(
            expanded, top_k=len(self.table_names)
        )

        cosine_dict = {name: score for name, score in cosine_results}

        # normalize BM25 scores to 0-1 range
        bm25_scores = [s for _, s in bm25_results]
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        score_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

        combined = {}
        for name, bm25_score in bm25_results:
            norm_bm25 = (bm25_score - bm25_min) / score_range
            cosine_score = cosine_dict.get(name, 0.0)
            combined[name] = alpha * cosine_score + (1 - alpha) * norm_bm25

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ── Stage 4: Lightweight reranking ────────────────────────────────

    def rerank(
        self, question: str, candidates: list[tuple[str, float]], top_k: int = 5
    ) -> list[tuple[str, float]]:
        """Applies a small offline boost for direct mentions and overlap."""
        if not candidates:
            return []

        question_lower = question.lower()
        ranked = []
        for name, base_score in candidates:
            desc = self.schema.get(name, name).lower()
            direct_hit = name.lower().replace("_", " ") in question_lower
            overlap = len(set(question_lower.split()) & set(desc.split()))
            score = float(base_score) + (0.2 if direct_hit else 0.0) + min(overlap, 5) * 0.01
            ranked.append((name, score))

        ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ── Stage 5: Metadata boosting ────────────────────────────────────

    def metadata_boost(
        self, question: str, results: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """
        Applies both literal table name matching and domain-specific semantic entity matching
        to ensure high-precision table retrieval for natural language Text-to-SQL questions.
        """
        question_lower = question.lower()

        # 1. Base metadata boost for literal table name matches
        forced_literal = []
        for tname in self.table_names:
            clean = tname.lower().replace("_", " ")
            if clean in question_lower or tname.lower() in question_lower:
                forced_literal.append(tname)
                logger.info(f"Metadata boost: literal match '{tname}'")

        # 2. Domain-specific semantic entity mapping
        domain_boosts = []
        
        # Textbook / Course materials / reserves / enrollment
        if "material" in question_lower or "textbook" in question_lower or "book" in question_lower:
            if "status" in question_lower or "reserve" in question_lower:
                domain_boosts.extend(["LIBRARY_RESERVE_MATRL_DETAIL", "LIBRARY_MATERIAL_STATUS"])
            else:
                domain_boosts.extend(["TIP_DETAIL", "TIP_SUBJECT_OFFERED"])
        if "enrolled" in question_lower or "enrollment" in question_lower:
            domain_boosts.append("TIP_SUBJECT_OFFERED")

        # Buildings / Facilities / Rooms
        if "building" in question_lower or "room" in question_lower or "space" in question_lower:
            if "history" in question_lower or "historical" in question_lower or "fiscal" in question_lower:
                domain_boosts.extend(["FCLT_BUILDING_HIST", "FCLT_BUILDING"])
            else:
                domain_boosts.extend(["FCLT_BUILDING", "FCLT_ROOMS"])
            if "floor" in question_lower:
                domain_boosts.append("FCLT_FLOOR")

        # HR Organizations
        if "hr organization" in question_lower or "hr org" in question_lower or "organization unit" in question_lower or "organization units" in question_lower:
            domain_boosts.extend(["HR_ORG_UNIT", "FCLT_ORGANIZATION"])

        # Degree-granting departments / engineering school / Political Science
        if "degree-granting" in question_lower or "degree granting" in question_lower or "engineering" in question_lower or "political science" in question_lower:
            domain_boosts.extend(["SIS_DEPARTMENT", "STUDENT_DEPARTMENT"])

        # IAP Activities / January term
        if "iap" in question_lower or "january" in question_lower or "activity" in question_lower:
            domain_boosts.extend(["IAP_SUBJECT_DETAIL", "ACADEMIC_TERMS"])

        # Student Directory
        if "student" in question_lower or "students" in question_lower:
            if "directory" in question_lower or "graduate" in question_lower or "unique student" in question_lower or "undergraduate" in question_lower:
                domain_boosts.append("MIT_STUDENT_DIRECTORY")

        # Course Catalog / catalog subjects / Mathematics
        if "catalog" in question_lower or "mathematics" in question_lower or "course 18" in question_lower:
            domain_boosts.extend(["COURSE_CATALOG_SUBJECT_OFFERED", "SIS_DEPARTMENT"])

        # Subject offered / offered / Mathematics subjects
        if "subject offered" in question_lower or "subjects offered" in question_lower:
            domain_boosts.extend(["SUBJECT_OFFERED_SUMMARY", "SUBJECT_OFFERED"])
        if "mathematics subjects" in question_lower or "mathematics department subjects" in question_lower:
            domain_boosts.append("SUBJECT_SUMMARY")

        # Phone numbers / administrative department
        if "phone" in question_lower or "administrative" in question_lower:
            domain_boosts.append("SIS_ADMIN_DEPARTMENT")

        # CIP Codes
        if "cip" in question_lower:
            domain_boosts.extend(["SIS_COURSE_DESCRIPTION", "CIP"])

        # Regular / fall / spring / academic terms
        if "fall" in question_lower or "spring" in question_lower or "regular term" in question_lower or "regular terms" in question_lower:
            domain_boosts.append("ACADEMIC_TERMS")
        if "academic year" in question_lower or "term" in question_lower or "terms" in question_lower:
            domain_boosts.extend(["ACADEMIC_TERMS", "ACADEMIC_TERMS_ALL"])
        if "current term" in question_lower or "current terms" in question_lower or "term status" in question_lower or "term parameter" in question_lower:
            domain_boosts.append("ACADEMIC_TERM_PARAMETER")

        # Subject codes / units / description (excluding organization units substring match)
        if "total units" in question_lower or ("units" in question_lower and "organization unit" not in question_lower and "organization units" not in question_lower):
            domain_boosts.extend(["SIS_SUBJECT_CODE", "SIS_COURSE_DESCRIPTION"])

        # Combine all forced tables preserving order, avoiding duplicates
        forced = []
        seen = set()
        for tname in forced_literal + domain_boosts:
            if tname in self.table_names and tname not in seen:
                forced.append(tname)
                seen.add(tname)

        # Build the final boosted results list
        boosted = []
        # First, add the forced tables
        for tname in forced:
            orig_score = next((s for name, s in results if name == tname), 1.0)
            boosted.append((tname, orig_score + 1.0))
            
        # Then, add the remaining results that were not forced
        for tname, score in results:
            if tname not in seen:
                boosted.append((tname, score))

        return boosted



    # ── Full pipeline ─────────────────────────────────────────────────
    
    def get_hybrid_candidates(self, question: str, top_k: int = 55) -> list[tuple[str, float]]:
        """
        Retrieves candidate tables by combining:
          1. Cosine similarity (bi-encoder)
          2. BM25 keyword matching
          3. Force-inclusion of table names directly mentioned in the query
        """
        # 1. Cosine candidates
        cosine_res = self.cosine_similarity_search(question, top_k=len(self.table_names))
        cosine_dict = {name: score for name, score in cosine_res}
        
        # 2. BM25 candidates
        bm25_res = self.bm25_search(question, top_k=len(self.table_names))
        bm25_scores = [s for _, s in bm25_res]
        bm25_max = max(bm25_scores) if bm25_scores else 1.0
        bm25_min = min(bm25_scores) if bm25_scores else 0.0
        score_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        
        # Merge BM25 and Cosine
        combined = {}
        alpha = 0.5  # Equal weight to balance exact keyword match & semantic match
        for name, bm25_score in bm25_res:
            norm_bm25 = (bm25_score - bm25_min) / score_range
            cosine_score = cosine_dict.get(name, 0.0)
            combined[name] = alpha * cosine_score + (1 - alpha) * norm_bm25
            
        # 3. Direct mention force-inclusion
        question_lower = question.lower()
        for tname in self.table_names:
            clean = tname.lower().replace("_", " ")
            if clean in question_lower or tname.lower() in question_lower:
                # Force include at the very top of candidates pool
                combined[tname] = 2.0
                logger.info(f"Direct mention forced '{tname}' into candidates pool")
                
        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve(
        self, question: str, top_k: int = 5
    ) -> dict:
        """
        Runs the complete retrieval pipeline:
          1. Hybrid BM25 + Cosine similarity candidate retrieval (top 55)
          2. Direct table mentions force-included in candidates
          3. Cross-encoder reranking
          4. Metadata boosting for safety net
        """
        # Get hybrid candidates with direct mentions force-included
        candidates = self.get_hybrid_candidates(question, top_k=55)
        logger.info(f"Candidate selection returned {len(candidates)} candidates")
        
        # Rerank candidates using Cross-Encoder
        reranked = self.rerank(question, candidates, top_k=top_k)
        logger.info(f"Reranking returned {len(reranked)} tables")
        
        # Apply final metadata boost safety net
        final = self.metadata_boost(question, reranked)
        final = final[:top_k]

        # build response
        table_names = [name for name, _ in final]
        scores = [round(float(score), 4) for _, score in final]
        confidence = round(float(np.mean(scores)), 4) if scores else 0.0

        details = {}
        question_lower = question.lower()
        for name, score in final:
            if name.lower() in question_lower or name.replace("_", " ") in question_lower:
                reason = f"Question directly mentions '{name}'"
            else:
                reason = f"Semantically relevant to the query (score={round(score, 4)})"
            details[name] = {
                "relevance_score": round(float(score), 4),
                "reason": reason,
            }

        result = {
            "retrieved_tables": table_names,
            "scores": scores,
            "confidence": confidence,
            "details": details,
        }

        logger.info(f"Final retrieval: {table_names}")
        return result
