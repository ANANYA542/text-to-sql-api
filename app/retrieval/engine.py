from __future__ import annotations

import logging
import math
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util as st_util

logger = logging.getLogger(__name__)

# SQL Keyword Expansions
SQL_EXPANSIONS = {
    "how many": ["COUNT", "GROUP", "BY"],
    "number of": ["COUNT", "GROUP", "BY"],
    "more than": ["HAVING", "COUNT"],
    "greater than": ["HAVING", "COUNT"],
    "at least": ["HAVING", "COUNT"],
    "average": ["AVG", "GROUP", "BY"],
    "mean": ["AVG", "GROUP", "BY"],
    "total": ["SUM", "GROUP", "BY"],
    "sum": ["SUM", "GROUP", "BY"],
    "rank": ["ORDER", "BY", "DESC", "LIMIT"],
    "top": ["ORDER", "BY", "DESC", "LIMIT"],
    "highest": ["ORDER", "BY", "DESC", "LIMIT"],
    "each": ["GROUP", "BY"],
    "per": ["GROUP", "BY"],
    "by": ["GROUP", "BY"],
    "exclude": ["WHERE", "NOT"],
    "not": ["WHERE", "NOT"],
    "except": ["WHERE", "NOT"],
}

class RetrievalEngine:
    def __init__(self, schema: dict[str, str]):
        self.schema = schema
        self.table_names = list(schema.keys())
        self.descriptions = [schema[t] for t in self.table_names]

        # 1. Clean descriptions for embedding (remove type and role noise)
        logger.info("Cleaning schema descriptions for embedding...")
        self.clean_descriptions = {
            tname: self.clean_desc_for_embedding(schema[tname])
            for tname in self.table_names
        }
        self.clean_desc_list = [self.clean_descriptions[t] for t in self.table_names]

        # 2. Parse relationship graph for score propagation
        logger.info("Parsing relationship graph from schema...")
        self.relations = self.parse_relations(schema)

        logger.info("Loading Sentence-Transformer Bi-Encoder model (BAAI/bge-small-en-v1.5)...")
        try:
            self.bi_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        except Exception as e:
            logger.warning(f"Could not load BAAI/bge-small-en-v1.5: {e}. Falling back to all-MiniLM-L6-v2.")
            self.bi_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

        logger.info("Precomputing and caching table embeddings...")
        self.table_embeddings = self.bi_encoder.encode(self.clean_desc_list, convert_to_tensor=True)

        tokenized = [desc.lower().split() for desc in self.clean_desc_list]
        self.bm25 = BM25Okapi(tokenized)

        logger.info("Loading Cross-Encoder model...")
        try:
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="mps")
            logger.info("Loaded Cross-Encoder on MPS (Apple GPU)")
        except Exception as e:
            logger.warning(f"Could not load Cross-Encoder on MPS: {e}. Falling back to CPU.")
            self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

        logger.info("RetrievalEngine loaded.")

    def clean_desc_for_embedding(self, desc: str) -> str:
        """
        Strips type and role noise from schema description to fit bi-encoder 256-token limit.
        Original: Table: SIS_DEPARTMENT. This table is useful for... Schema structure: [COL1 (TYPE) [role: ROLE], ...]. Related Tables...
        Cleaned: Table: SIS_DEPARTMENT. This table is useful for... Columns: COL1, COL2, ...
        """
        table_part = ""
        usecase_part = ""
        columns_part = ""
        
        if "Table: " in desc:
            table_part = desc.split(".")[0] + "." # "Table: TABLE_NAME."
            
        if "Schema structure: [" in desc:
            usecase_part = desc.split("Schema structure: [")[0].replace(table_part, "").strip()
            cols_raw = desc.split("Schema structure: [")[1].split("]")[0]
            clean_cols = []
            for col_raw in cols_raw.split(","):
                col_name = col_raw.split("(")[0].strip()
                if col_name:
                    clean_cols.append(col_name)
            columns_part = "Columns: " + ", ".join(clean_cols)
        else:
            return desc
            
        return f"{table_part} {usecase_part} {columns_part}"

    def parse_relations(self, schema: dict[str, str]) -> dict[str, list[str]]:
        """Parses Related Tables from the enriched schema description."""
        relations = {}
        for name, desc in schema.items():
            name_upper = name.upper()
            rel_list = []
            if "Related Tables (Join Paths): [" in desc:
                rel_part = desc.split("Related Tables (Join Paths): [")[1].split("]")[0]
                for item in rel_part.split(","):
                    if " (via " in item:
                        rel_table = item.split(" (via ")[0].strip()
                        rel_list.append(rel_table.upper())
            relations[name_upper] = rel_list
        return relations

    def get_table_name_boosts(self, question: str) -> dict[str, float]:
        question_lower = question.lower()
        boosts = {}
        
        words = set(question_lower.split())
        clean_words = {w.strip("?,.()_:") for w in words}
        
        for tname in self.table_names:
            tname_lower = tname.lower()
            
            # Exact direct match of table name or space-separated table name
            if tname_lower in question_lower or tname_lower.replace("_", " ") in question_lower:
                boosts[tname] = 2.0
                continue
                
            # Prefix-stripped match
            parts = tname_lower.split("_")
            if len(parts) > 1:
                stripped = parts[1]
                # Check singular and plural forms
                if (stripped in clean_words or (stripped + "s") in clean_words or (stripped[:-1] in clean_words if stripped.endswith("s") else False)) and len(stripped) > 3:
                    boosts[tname] = 1.5
                    continue
                
                # e.g., FCLT_BUILDING_HIST -> building hist / building history
                if stripped == "building" and "hist" in parts:
                    if "building" in clean_words and any(w in clean_words for w in ("history", "historical", "hist")):
                        boosts[tname] = 1.8
                        continue
        return boosts

    def get_category_boosts(self, question: str) -> dict[str, float]:
        question_lower = question.lower()
        boosts = {}
        
        category_map = {
            "TIP_": ["textbook", "reading", "reserve", "book", "material", "reserves"],
            "FCLT_": ["building", "room", "space", "facilities", "floor", "site", "latitude", "longitude", "assignable"],
            "FAC_": ["building", "room", "space", "facilities", "floor", "site"],
            "HR_": ["employee", "organization unit", "hr org", "job", "salary", "hierarchy"],
            "SIS_": ["student", "department", "enrollment", "course", "subject", "academic", "class", "faculty", "gpa", "grade"],
        }
        
        for prefix, keywords in category_map.items():
            if any(kw in question_lower for kw in keywords):
                for tname in self.table_names:
                    if tname.startswith(prefix):
                        boosts[tname] = boosts.get(tname, 0.0) + 0.3
                        
        return boosts

    def expand_query(self, question: str) -> str:
        """Applies basic SQL expansions and translates abbreviations dynamically for BM25 search."""
        question_lower = question.lower()
        
        query_words = set()
        for w in question_lower.split():
            clean_w = w.strip("?,.()_")
            if clean_w:
                query_words.add(clean_w)
                if clean_w.endswith("s") and len(clean_w) > 4:
                    query_words.add(clean_w[:-1])

        extras = set()

        # SQL keyword expansions
        for phrase, keywords in SQL_EXPANSIONS.items():
            if phrase in question_lower:
                extras.update(keywords)

        stop_words = {
            "and", "of", "for", "the", "in", "to", "with", "within", "related", 
            "queries", "concerning", "stores", "records", "about", "each", 
            "what", "from", "their", "only", "than", "more", "less", "at", 
            "least", "have", "offered", "by", "that", "this", "table", "useful", "academic"
        }

        generic_trigger_words = {
            "course", "courses", "subject", "subjects", "academic", "university", 
            "record", "records", "system", "details", "summary", "metrics", "views",
            "timeline", "timelines", "tracking", "archive", "school", "schools",
            "term", "terms", "date", "dates", "time", "times"
        }

        # Dynamic abbreviation mapping using ABBREV_MAP from schema_loader
        from app.retrieval.schema_loader import ABBREV_MAP
        for abbrev, meaning in ABBREV_MAP.items():
            meaning_clean = meaning.replace(",", "").replace(".", "").lower()
            meaning_words = set(meaning_clean.split())
            
            meaning_words_filtered = {w for w in meaning_words if w not in stop_words and len(w) > 3}
            query_words_filtered = {w for w in query_words if w not in stop_words}
            
            meaning_words_trigger = {w for w in meaning_words_filtered if w not in generic_trigger_words}
            query_words_trigger = {w for w in query_words_filtered if w not in generic_trigger_words}
            
            if abbrev.lower() in query_words or query_words_trigger.intersection(meaning_words_trigger):
                extras.add(abbrev)
                extras.update(w.upper() for w in meaning_words_filtered)

        for tname in self.table_names:
            clean = tname.lower().replace("_", " ")
            if clean in question_lower or tname.lower() in question_lower:
                extras.add(tname)

        expanded = question + " " + " ".join(extras)
        return expanded

    def get_hybrid_candidates(self, question: str, top_k: int = 15) -> list[tuple[str, float]]:
        expanded = self.expand_query(question)
        
        # 1. BM25 Search
        bm25_scores = self.bm25.get_scores(expanded.lower().split())
        bm25_max = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        bm25_min = min(bm25_scores) if len(bm25_scores) > 0 else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        
        # 2. Cosine Similarity (using clean query to avoid bi-encoder noise)
        q_embedding = self.bi_encoder.encode(question, convert_to_tensor=True)
        cosine_scores = st_util.cos_sim(q_embedding, self.table_embeddings)[0].tolist()
        
        # 3. Combine scores
        combined = {}
        alpha = 0.5
        for idx, tname in enumerate(self.table_names):
            norm_bm25 = (bm25_scores[idx] - bm25_min) / bm25_range
            cos_score = cosine_scores[idx]
            combined[tname] = alpha * cos_score + (1 - alpha) * norm_bm25
            
        # 4. Add Name Match & Category Boosts
        name_boosts = self.get_table_name_boosts(question)
        cat_boosts = self.get_category_boosts(question)
        for tname in self.table_names:
            combined[tname] = combined.get(tname, 0.0) + name_boosts.get(tname, 0.0) + cat_boosts.get(tname, 0.0)
            
        # 5. Schema Graph Score Propagation
        propagated = combined.copy()
        for tname, score in combined.items():
            if score > 1.0:
                neighbors = self.relations.get(tname.upper(), [])
                for neighbor in neighbors:
                    if neighbor in propagated:
                        propagated[neighbor] = max(propagated[neighbor], score * 0.4)
                        
        ranked = sorted(propagated.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    def retrieve(self, question: str, top_k: int = 5) -> dict:
        # 1. Get 15 hybrid candidates (aggressively reduced from 35/55 to improve speed and quality)
        candidates = self.get_hybrid_candidates(question, top_k=15)
        
        if not candidates:
            return {"retrieved_tables": [], "scores": [], "confidence": 0.0, "details": {}}
            
        # 2. Prepare Cross-Encoder pairs
        pairs = []
        for name, _ in candidates:
            desc = self.schema.get(name, name)
            clean_desc = self.clean_descriptions.get(name, desc)
            pairs.append([question, clean_desc])
            
        raw_logits = np.asarray(self.cross_encoder.predict(pairs), dtype=float)
        raw_logits = np.nan_to_num(raw_logits, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 3. Fusion scoring
        fusion_results = []
        prob_dict = {}
        for i, (name, cand_score) in enumerate(candidates):
            logit = float(raw_logits[i])
            prob = 1.0 / (1.0 + math.exp(-logit))
            prob_dict[name] = prob
            
            # Fuse candidate score (keyword/bi-encoder/graph) with CE probability
            fusion_score = cand_score + prob * 1.5
            fusion_results.append((name, fusion_score))
            
        # Sort by fusion score
        ranked = sorted(fusion_results, key=lambda x: x[1], reverse=True)
        final_candidates = ranked[:top_k]
        
        table_names = [name for name, _ in final_candidates]
        scores = [round(prob_dict[name], 4) for name in table_names]
        confidence = scores[0] if scores else 0.0
        
        details = {}
        question_lower = question.lower()
        for name in table_names:
            prob = round(prob_dict[name], 4)
            clean_name = name.lower().replace("_", " ")
            if clean_name in question_lower or name.lower() in question_lower:
                reason = f"Question directly mentions table '{name}'"
            else:
                reason = f"Semantically mapped table based on query intent (score={prob})"
                
            details[name] = {
                "relevance_score": prob,
                "reason": reason,
                "raw_relevance": prob,
            }
            
        return {
            "retrieved_tables": table_names,
            "scores": scores,
            "confidence": confidence,
            "details": details,
        }
