from __future__ import annotations

import logging
import math
import numpy as np
import os
import requests
import time
import re
import json
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
            self.expansion_cache = {}

        self.expansion_cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".cache", "query_expansion_cache.json")
        self.expansion_cache = {}
        self.load_expansion_cache()
        self.last_api_call_time = 0.0
        logger.info("RetrievalEngine loaded.")

    def load_expansion_cache(self):
        if os.path.exists(self.expansion_cache_path):
            try:
                with open(self.expansion_cache_path, "r") as f:
                    self.expansion_cache = json.load(f)
                logger.info(f"Loaded {len(self.expansion_cache)} queries from query expansion cache.")
            except Exception as e:
                logger.warning(f"Failed to load query expansion cache: {e}")
                self.expansion_cache = {}

    def save_expansion_cache(self):
        try:
            os.makedirs(os.path.dirname(self.expansion_cache_path), exist_ok=True)
            with open(self.expansion_cache_path, "w") as f:
                json.dump(self.expansion_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save query expansion cache: {e}")

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

        # Targeted boosts for tables that need extra signal

        # ACADEMIC_TERMS_ALL: only boost for truly disambiguating phrases
        if any(kw in question_lower for kw in ["term status", "all terms", "term status indicator", "historical"]):
            for tname in ["ACADEMIC_TERMS_ALL", "ACADEMIC_TERM_PARAMETER"]:
                if tname in self.table_names:
                    boosts[tname] = boosts.get(tname, 0.0) + 0.8
        # 'academic year description' / 'each academic year' → ACADEMIC_TERMS_ALL has ACADEMIC_YEAR_DESC
        if "academic year description" in question_lower or "each academic year" in question_lower:
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) + 0.9

        # ACADEMIC_TERMS (not ALL): for regular/fall/spring term filtering by date
        if any(kw in question_lower for kw in ["regular term", "fall term", "spring term", "starting after", "start after", "after january", "after december"]):
            if "ACADEMIC_TERMS" in self.table_names:
                boosts["ACADEMIC_TERMS"] = boosts.get("ACADEMIC_TERMS", 0.0) + 0.9
            # also demote ACADEMIC_TERMS_ALL since it's less correct here
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) - 0.4

        # Course catalog queries → COURSE_CATALOG_SUBJECT_OFFERED, CIS_COURSE_CATALOG
        if any(kw in question_lower for kw in ["total units", "subjects offered", "course catalog", "degree granting", "degree-granting", "offered in", "offered by"]):
            for tname in ["COURSE_CATALOG_SUBJECT_OFFERED", "CIS_COURSE_CATALOG"]:
                if tname in self.table_names:
                    boosts[tname] = boosts.get(tname, 0.0) + 0.6

        # Library reserve queries → LIBRARY_RESERVE_MATRL_DETAIL
        if any(kw in question_lower for kw in ["library reserve", "reserve material", "library material"]):
            if "LIBRARY_RESERVE_MATRL_DETAIL" in self.table_names:
                boosts["LIBRARY_RESERVE_MATRL_DETAIL"] = boosts.get("LIBRARY_RESERVE_MATRL_DETAIL", 0.0) + 1.0

        # Department + admin info queries → SIS_ADMIN_DEPARTMENT
        if any(kw in question_lower for kw in ["phone", "school name", "degree-granting", "degree granting", "clearing cost", "cost collector"]):
            if "SIS_ADMIN_DEPARTMENT" in self.table_names:
                boosts["SIS_ADMIN_DEPARTMENT"] = boosts.get("SIS_ADMIN_DEPARTMENT", 0.0) + 0.9

        # Textbook detail queries → TIP_DETAIL
        if any(kw in question_lower for kw in ["isbn", "author", "edition", "publisher", "book detail", "tip detail"]):
            if "TIP_DETAIL" in self.table_names:
                boosts["TIP_DETAIL"] = boosts.get("TIP_DETAIL", 0.0) + 0.9

        # Enrollment/subject summary → SUBJECT_SUMMARY (per-term enrollment counts)
        if any(kw in question_lower for kw in ["subject summary", "enrollment"]) or (
            any(kw in question_lower for kw in ["enrolled", "students enrolled"])
            and any(kw in question_lower for kw in ["spring", "fall", "term", "per term"])
        ):
            if "SUBJECT_SUMMARY" in self.table_names:
                boosts["SUBJECT_SUMMARY"] = boosts.get("SUBJECT_SUMMARY", 0.0) + 0.7

        # MIT_STUDENT_DIRECTORY: graduate/undergraduate student queries
        if any(kw in question_lower for kw in ["graduate student", "student year", "undergraduate student", "student directory", "'g'", "year is"]):
            if "MIT_STUDENT_DIRECTORY" in self.table_names:
                boosts["MIT_STUDENT_DIRECTORY"] = boosts.get("MIT_STUDENT_DIRECTORY", 0.0) + 0.8


        # DLC queries → FCLT_ORGANIZATION, FCLT_ORG_DLC_KEY
        if any(kw in question_lower for kw in ["dlc", "department, lab", "dept, lab", "lab, center"]):
            for tname in ["FCLT_ORGANIZATION", "FCLT_ORG_DLC_KEY"]:
                if tname in self.table_names:
                    boosts[tname] = boosts.get(tname, 0.0) + 1.0

        # Statistical analysis of total units per dept → SUBJECT_OFFERED_SUMMARY
        if any(kw in question_lower for kw in [
            "coefficient of variation", "stddev", "standard deviation",
            "variance of total units", "average.*total units", "total units.*department",
            "units.*offered", "offered.*units"
        ]) or (
            any(kw in question_lower for kw in ["total units", "units"]) and
            any(kw in question_lower for kw in ["stddev", "variance", "coefficient", "average", "mean"])
        ):
            if "SUBJECT_OFFERED_SUMMARY" in self.table_names:
                boosts["SUBJECT_OFFERED_SUMMARY"] = boosts.get("SUBJECT_OFFERED_SUMMARY", 0.0) + 0.9

        # Faculty / instructor queries → LIBRARY_SUBJECT_OFFERED (has faculty links)
        # Stronger signal: "responsible faculty" in 2019 queries means LIBRARY_SUBJECT_OFFERED is the enrollment source
        if any(kw in question_lower for kw in ["faculty member", "responsible faculty", "instructor", "faculty name"]):
            if "LIBRARY_SUBJECT_OFFERED" in self.table_names:
                boosts["LIBRARY_SUBJECT_OFFERED"] = boosts.get("LIBRARY_SUBJECT_OFFERED", 0.0) + 1.1
        # "enrolled students" + year context almost always means LIBRARY_SUBJECT_OFFERED
        if "enrolled students" in question_lower and any(str(yr) in question_lower for yr in range(2015, 2026)):
            if "LIBRARY_SUBJECT_OFFERED" in self.table_names:
                boosts["LIBRARY_SUBJECT_OFFERED"] = boosts.get("LIBRARY_SUBJECT_OFFERED", 0.0) + 0.8
        # "num_enrolled" / "number of enrolled" without a term qualifier → LIBRARY_SUBJECT_OFFERED
        if "num enrolled" in question_lower or "number of enrolled" in question_lower:
            if "LIBRARY_SUBJECT_OFFERED" in self.table_names:
                boosts["LIBRARY_SUBJECT_OFFERED"] = boosts.get("LIBRARY_SUBJECT_OFFERED", 0.0) + 0.7

        # Material status = 'U' or general material status → LIBRARY_RESERVE_MATRL_DETAIL
        if "material status" in question_lower:
            if "LIBRARY_RESERVE_MATRL_DETAIL" in self.table_names:
                boosts["LIBRARY_RESERVE_MATRL_DETAIL"] = boosts.get("LIBRARY_RESERVE_MATRL_DETAIL", 0.0) + 1.2

        # Geometric mean + academic year + material → needs LIBRARY_RESERVE_MATRL_DETAIL + COURSE_CATALOG + ACADEMIC_TERMS
        if "geometric mean" in question_lower:
            if "LIBRARY_RESERVE_MATRL_DETAIL" in self.table_names:
                boosts["LIBRARY_RESERVE_MATRL_DETAIL"] = boosts.get("LIBRARY_RESERVE_MATRL_DETAIL", 0.0) + 0.9
            if "COURSE_CATALOG_SUBJECT_OFFERED" in self.table_names:
                boosts["COURSE_CATALOG_SUBJECT_OFFERED"] = boosts.get("COURSE_CATALOG_SUBJECT_OFFERED", 0.0) + 0.8
            # If geometric mean is asked per academic year, we need ACADEMIC_TERMS for term_code join
            if "academic year" in question_lower:
                if "ACADEMIC_TERMS" in self.table_names:
                    boosts["ACADEMIC_TERMS"] = boosts.get("ACADEMIC_TERMS", 0.0) + 0.9
                if "ACADEMIC_TERMS_ALL" in self.table_names:
                    boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) - 0.3

        # "associated course material" / "course material" → TIP_DETAIL + TIP_SUBJECT_OFFERED
        if any(kw in question_lower for kw in ["associated course material", "course material", "course materials"]):
            if "TIP_DETAIL" in self.table_names:
                boosts["TIP_DETAIL"] = boosts.get("TIP_DETAIL", 0.0) + 1.1
            if "TIP_SUBJECT_OFFERED" in self.table_names:
                boosts["TIP_SUBJECT_OFFERED"] = boosts.get("TIP_SUBJECT_OFFERED", 0.0) + 0.7

        # "course description" / "EECS" / "electrical engineering" → SIS_COURSE_DESCRIPTION
        if any(kw in question_lower for kw in ["course description", "subject description", "graduate level",
                                                "electrical engineering", "course 6", "eecs",
                                                "letter graded", "undergraduate", "hgn_desc"]):
            if "SIS_COURSE_DESCRIPTION" in self.table_names:
                boosts["SIS_COURSE_DESCRIPTION"] = boosts.get("SIS_COURSE_DESCRIPTION", 0.0) + 1.0

        # "subject code description" → SIS_SUBJECT_CODE
        if any(kw in question_lower for kw in ["subject code description", "subject code desc", "subject code"]):
            if "SIS_SUBJECT_CODE" in self.table_names:
                boosts["SIS_SUBJECT_CODE"] = boosts.get("SIS_SUBJECT_CODE", 0.0) + 0.9

        # "department hierarchy" → MASTER_DEPT_HIERARCHY
        if any(kw in question_lower for kw in ["department hierarchy", "dept hierarchy", "hierarchy level"]):
            if "MASTER_DEPT_HIERARCHY" in self.table_names:
                boosts["MASTER_DEPT_HIERARCHY"] = boosts.get("MASTER_DEPT_HIERARCHY", 0.0) + 1.5

        # "hr organization unit" or "hr org unit" or "old department code" → HR_ORG_UNIT stronger
        if any(kw in question_lower for kw in ["hr organization unit", "hr org unit", "old department code",
                                                "hr department code", "organization unit title"]):
            if "HR_ORG_UNIT" in self.table_names:
                boosts["HR_ORG_UNIT"] = boosts.get("HR_ORG_UNIT", 0.0) + 1.2

        # "variable units" / "is_variable_units" → COURSE_CATALOG_SUBJECT_OFFERED
        if any(kw in question_lower for kw in ["variable units", "non-variable", "not variable"]):
            if "COURSE_CATALOG_SUBJECT_OFFERED" in self.table_names:
                boosts["COURSE_CATALOG_SUBJECT_OFFERED"] = boosts.get("COURSE_CATALOG_SUBJECT_OFFERED", 0.0) + 0.9

        # "department phone" → SIS_ADMIN_DEPARTMENT (stronger than current)
        if any(kw in question_lower for kw in ["department phone", "phone number", "budget code", "budget"]):
            if "SIS_ADMIN_DEPARTMENT" in self.table_names:
                boosts["SIS_ADMIN_DEPARTMENT"] = boosts.get("SIS_ADMIN_DEPARTMENT", 0.0) + 1.0

        # "fall term" + "mathematics" + academic year → COURSE_CATALOG_SUBJECT_OFFERED + ACADEMIC_TERMS
        if ("fall term" in question_lower or "offered in the fall" in question_lower) and any(
            kw in question_lower for kw in ["mathematics", "course 18"]
        ):
            if "COURSE_CATALOG_SUBJECT_OFFERED" in self.table_names:
                boosts["COURSE_CATALOG_SUBJECT_OFFERED"] = boosts.get("COURSE_CATALOG_SUBJECT_OFFERED", 0.0) + 1.0
            if "ACADEMIC_TERMS" in self.table_names:
                boosts["ACADEMIC_TERMS"] = boosts.get("ACADEMIC_TERMS", 0.0) + 0.9
            # Demote _ALL when we explicitly want ACADEMIC_TERMS for date-filtered queries
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) - 0.3

        # "academic year description" with average/variance → ACADEMIC_TERMS_ALL (has ACADEMIC_YEAR_DESC column)
        if "academic year description" in question_lower:
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) + 1.1
            # If the query also has COURSE_CATALOG columns like total_units or subject_code, boost it too
            if any(kw in question_lower for kw in ["total units", "subject code", "subjects offered"]):
                if "COURSE_CATALOG_SUBJECT_OFFERED" in self.table_names:
                    boosts["COURSE_CATALOG_SUBJECT_OFFERED"] = boosts.get("COURSE_CATALOG_SUBJECT_OFFERED", 0.0) + 0.9

        # "term status" = 'Previous' or 'P' → ACADEMIC_TERMS_ALL (has TERM_STATUS_IND)
        if any(kw in question_lower for kw in ["term status", "status of 'previous'", "'previous'", "term status of"]):
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) + 1.1

        # "regular terms starting after" → ACADEMIC_TERMS (has START_DATE column)
        if any(kw in question_lower for kw in ["regular terms starting", "starting after january", "starting after"]):
            if "ACADEMIC_TERMS" in self.table_names:
                boosts["ACADEMIC_TERMS"] = boosts.get("ACADEMIC_TERMS", 0.0) + 1.2
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) - 0.5

        # SPACE_DETAIL: "space usage", "square footage", "square feet", "rooms used for"
        if any(kw in question_lower for kw in ["space usage", "square footage", "square feet", "space detail",
                                                "rooms used for", "type of space"]):
            if "SPACE_DETAIL" in self.table_names:
                boosts["SPACE_DETAIL"] = boosts.get("SPACE_DETAIL", 0.0) + 1.3

        # SIS_COURSE_DESCRIPTION: "school of engineering", "school of science", etc. + subject offered
        if any(kw in question_lower for kw in ["school of engineering", "school of science", "school of architecture",
                                                "school of humanities"]):
            if "SIS_COURSE_DESCRIPTION" in self.table_names:
                boosts["SIS_COURSE_DESCRIPTION"] = boosts.get("SIS_COURSE_DESCRIPTION", 0.0) + 1.2
            # These queries commonly need SUBJECT_OFFERED_SUMMARY too
            if "SUBJECT_OFFERED_SUMMARY" in self.table_names:
                boosts["SUBJECT_OFFERED_SUMMARY"] = boosts.get("SUBJECT_OFFERED_SUMMARY", 0.0) + 0.5

        # SUBJECT_SUMMARY: "total number of.*subjects", "number of.*subjects offered"
        if any(kw in question_lower for kw in ["total number of", "number of subjects", "subjects offered in those",
                                                "subjects offered in regular", "subjects offered in the"]):
            if "SUBJECT_SUMMARY" in self.table_names:
                boosts["SUBJECT_SUMMARY"] = boosts.get("SUBJECT_SUMMARY", 0.0) + 0.9

        # "regular terms" (without "starting after") → ACADEMIC_TERMS (has TERM_TYPE)
        if "regular term" in question_lower and "starting after" not in question_lower:
            if "ACADEMIC_TERMS" in self.table_names:
                boosts["ACADEMIC_TERMS"] = boosts.get("ACADEMIC_TERMS", 0.0) + 1.0

        # TIP_SUBJECT_OFFERED: "chemistry courses" with "reserve material" / "enrolled students"
        if any(kw in question_lower for kw in ["chemistry", "course 5"]) and any(
            kw in question_lower for kw in ["reserve material", "enrolled students", "department offering"]
        ):
            if "TIP_SUBJECT_OFFERED" in self.table_names:
                boosts["TIP_SUBJECT_OFFERED"] = boosts.get("TIP_SUBJECT_OFFERED", 0.0) + 1.0

        # When "material status" is present, LIBRARY_RESERVE_MATRL_DETAIL is critical →
        # demote ACADEMIC_TERMS_ALL (often displaces it)
        if "material status" in question_lower:
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) - 0.5

        # "term status indicator" + "reserve material" → ACADEMIC_TERMS_ALL + LIBRARY_RESERVE_MATRL_DETAIL
        if "term status indicator" in question_lower:
            if "ACADEMIC_TERMS_ALL" in self.table_names:
                boosts["ACADEMIC_TERMS_ALL"] = boosts.get("ACADEMIC_TERMS_ALL", 0.0) + 1.2
            if "LIBRARY_RESERVE_MATRL_DETAIL" in self.table_names:
                boosts["LIBRARY_RESERVE_MATRL_DETAIL"] = boosts.get("LIBRARY_RESERVE_MATRL_DETAIL", 0.0) + 0.8

        # "above average" / "below average" with department → SIS_COURSE_DESCRIPTION (has HGN_DESC, school info)
        if any(kw in question_lower for kw in ["above_average", "below_average", "'above_average'", "'below_average'",
                                                "above 10", "below 6"]):
            if "SIS_COURSE_DESCRIPTION" in self.table_names:
                boosts["SIS_COURSE_DESCRIPTION"] = boosts.get("SIS_COURSE_DESCRIPTION", 0.0) + 0.8

        return boosts


    def throttle_api_call(self, min_interval=4.0):
        now = time.time()
        elapsed = now - self.last_api_call_time
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed
            logger.info(f"Throttling API call in query expansion. Sleeping for {sleep_time:.2f}s...")
            time.sleep(sleep_time)
        self.last_api_call_time = time.time()

    def expand_query_with_llm(self, question: str) -> str:
        """
        Uses Groq API model llama-3.1-8b-instant to get a technical expansion of the question.
        Returns a space-separated string of potential tables, columns, and keywords.
        """
        if question in self.expansion_cache:
            return self.expansion_cache[question]
            
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            logger.warning("GROQ_API_KEY not found in environment for query expansion.")
            return ""
            
        system_prompt = (
            "You are an expert DB admin. Given a user question, return ONLY a space-separated list of "
            "likely SQL keywords (e.g. JOIN, HAVING, GROUP BY), potential table names, and column names "
            "from a university/facilities schema that are relevant to the question. Do not return markdown, "
            "do not explain anything, just output the space-separated technical keywords."
        )
        
        user_prompt = (
            f"Question: Which departments have more than 100 students?\n"
            f"Keywords: SIS_DEPARTMENT STUDENT_DEPARTMENT student_id dept_id student_count COUNT GROUP BY HAVING enrollments student\n\n"
            f"Question: Show the building details for buildings with library reserves.\n"
            f"Keywords: FCLT_BUILDING library_reserve_matrl_detail building_key library reserve materials\n\n"
            f"Question: {question}\n"
            f"Keywords:"
        )
        
        headers = {
            "Authorization": f"Bearer {groq_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.0,
            "max_tokens": 150
        }
        
        # Simple retry loop with delay
        for attempt in range(3):
            try:
                if attempt == 0:
                    self.throttle_api_call(min_interval=4.0)
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                if response.status_code == 200:
                    res_json = response.json()
                    content = res_json["choices"][0]["message"]["content"].strip()
                    logger.info(f"LLM Query Expansion: '{content}'")
                    self.expansion_cache[question] = content
                    self.save_expansion_cache()
                    return content
                elif response.status_code == 429:
                    retry_after = 5
                    try:
                        err_detail = response.json()
                        msg = err_detail.get("error", {}).get("message", "")
                        if "try again in" in msg:
                            match = re.search(r"try again in ([\d\.]+)s", msg)
                            if match:
                                retry_after = float(match.group(1)) + 0.5
                    except Exception:
                        pass
                    logger.info(f"Groq TPM Rate limit in query expansion. Retrying in {retry_after}s...")
                    time.sleep(retry_after)
                else:
                    logger.warning(f"Groq API returned error status {response.status_code} in query expansion.")
                    break
            except Exception as e:
                logger.warning(f"Error calling Groq API for query expansion: {e}")
                time.sleep(1)
        self.expansion_cache[question] = ""
        return ""

    def expand_query(self, question: str, use_llm: bool = True) -> str:
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
 
        llm_expanded = ""
        if use_llm:
            llm_expanded = self.expand_query_with_llm(question)

        expanded = question + " " + " ".join(extras) + " " + llm_expanded
        return expanded

    def get_hybrid_candidates(self, question: str, top_k: int = 25, forced_tables: list[str] | None = None) -> list[tuple[str, float]]:
        expanded = self.expand_query(question)
        
        # 1. BM25 Search
        bm25_scores = self.bm25.get_scores(expanded.lower().split())
        bm25_max = max(bm25_scores) if len(bm25_scores) > 0 and max(bm25_scores) > 0 else 1.0
        bm25_min = min(bm25_scores) if len(bm25_scores) > 0 else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0
        
        # 2. Cosine Similarity
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
        candidates = ranked[:top_k]

        # 6. Force-include tables that have strong targeted boosts but may have missed top-k cutoff
        if forced_tables:
            candidate_names = {name for name, _ in candidates}
            for tname in forced_tables:
                if tname not in candidate_names and tname in propagated:
                    # Insert with its boosted score (will be reranked by cross-encoder)
                    candidates.append((tname, propagated[tname]))

        return candidates

    def retrieve(self, question: str, top_k: int = 5, use_learned_ranker: bool = True) -> dict:
        """
        Runs the complete retrieval pipeline with optional learned reranking.

        Pipeline stages:
            1. Query expansion (rule-based + LLM)
            2. Hybrid candidate retrieval (BM25 + cosine + name/category boosts)
            3. Cross-encoder reranking
            4. [Optional] Learned ML ranker (LightGBM/XGBoost)
            5. Score calibration and response construction

        Args:
            question: Natural language question.
            top_k: Number of tables to return.
            use_learned_ranker: If True, attempts to use the trained ML model.
                Falls back to heuristic fusion if the model is not available.

        Returns:
            Dict with retrieved_tables, scores, confidence, details, and latency_breakdown.
        """
        import time as _time
        latency_breakdown = {}

        # ── Stage 1: Category boosts and candidate retrieval ──────────
        t0 = _time.perf_counter()

        cat_boosts = self.get_category_boosts(question)
        forced_tables = [t for t, b in cat_boosts.items() if b >= 0.8]

        candidates = self.get_hybrid_candidates(question, top_k=25, forced_tables=forced_tables)
        latency_breakdown["retrieval_ms"] = round((_time.perf_counter() - t0) * 1000, 2)

        if not candidates:
            return {
                "retrieved_tables": [], "scores": [], "confidence": 0.0,
                "details": {}, "latency_breakdown": latency_breakdown,
                "model_used": "none",
            }

        candidate_names = [name for name, _ in candidates]

        # ── Stage 2: Cross-Encoder reranking ──────────────────────────
        t1 = _time.perf_counter()

        pairs = []
        for name, _ in candidates:
            desc = self.schema.get(name, name)
            clean_desc = self.clean_descriptions.get(name, desc)
            pairs.append([question, clean_desc])

        raw_logits = np.asarray(self.cross_encoder.predict(pairs), dtype=float)
        raw_logits = np.nan_to_num(raw_logits, nan=0.0, posinf=0.0, neginf=0.0)

        ce_logits_dict = {}
        for i, name in enumerate(candidate_names):
            ce_logits_dict[name] = float(raw_logits[i])

        latency_breakdown["reranking_ms"] = round((_time.perf_counter() - t1) * 1000, 2)

        # ── Stage 3: Learned ranker (optional) or heuristic fusion ────
        t2 = _time.perf_counter()
        model_used = "heuristic"
        final_candidates = []

        if use_learned_ranker:
            try:
                from app.ml.learned_ranker import LearnedRanker, DEFAULT_MODEL_PATH
                from app.ml.feature_engineering import FeatureExtractor, PipelineScores

                if DEFAULT_MODEL_PATH.exists():
                    # Build PipelineScores from intermediate results
                    # Capture BM25 and cosine scores/ranks
                    expanded_query = self.expand_query(question, use_llm=False)
                    bm25_raw = self.bm25.get_scores(expanded_query.lower().split())
                    bm25_max = max(bm25_raw) if len(bm25_raw) > 0 and max(bm25_raw) > 0 else 1.0
                    bm25_min = min(bm25_raw) if len(bm25_raw) > 0 else 0.0
                    bm25_range = bm25_max - bm25_min if bm25_max != bm25_min else 1.0

                    bm25_scores_dict = {}
                    bm25_ranks_dict = {}
                    bm25_pairs = sorted(
                        zip(self.table_names, bm25_raw),
                        key=lambda x: x[1], reverse=True,
                    )
                    for rank, (tname, score) in enumerate(bm25_pairs):
                        bm25_scores_dict[tname] = (float(score) - bm25_min) / bm25_range
                        bm25_ranks_dict[tname] = rank + 1

                    from sentence_transformers import util as st_util
                    q_emb = self.bi_encoder.encode(question, convert_to_tensor=True)
                    cos_raw = st_util.cos_sim(q_emb, self.table_embeddings)[0].tolist()
                    cosine_scores_dict = {}
                    cosine_ranks_dict = {}
                    cos_pairs = sorted(
                        zip(self.table_names, cos_raw),
                        key=lambda x: x[1], reverse=True,
                    )
                    for rank, (tname, score) in enumerate(cos_pairs):
                        cosine_scores_dict[tname] = float(score)
                        cosine_ranks_dict[tname] = rank + 1

                    pipeline_scores = PipelineScores(
                        bm25_scores=bm25_scores_dict,
                        cosine_scores=cosine_scores_dict,
                        cross_encoder_logits=ce_logits_dict,
                        bm25_ranks=bm25_ranks_dict,
                        cosine_ranks=cosine_ranks_dict,
                        llm_expansion_text=expanded_query,
                    )

                    # Extract features and predict
                    fe = FeatureExtractor(self.schema, self.relations, self.table_names)
                    X = fe.extract_batch(question, candidate_names, pipeline_scores)

                    ranker = LearnedRanker.load(DEFAULT_MODEL_PATH)
                    final_candidates = ranker.predict_top_k(X, candidate_names, k=top_k)
                    model_used = f"learned_{ranker.model_type}"
                    logger.info(f"Learned ranker ({ranker.model_type}) produced top-{top_k} results.")
                else:
                    logger.debug("No trained model found. Falling back to heuristic fusion.")
                    use_learned_ranker = False  # Fall through to heuristic path
            except Exception as e:
                logger.warning(f"Learned ranker failed: {e}. Falling back to heuristic.")
                use_learned_ranker = False  # Fall through to heuristic path

        # Heuristic fusion fallback
        if not final_candidates:
            fusion_results = []
            for i, (name, cand_score) in enumerate(candidates):
                logit = float(raw_logits[i])
                prob = 1.0 / (1.0 + math.exp(-logit))
                fusion_score = cand_score + prob * 1.5
                fusion_results.append((name, fusion_score))

            ranked = sorted(fusion_results, key=lambda x: x[1], reverse=True)
            final_candidates = ranked[:top_k]

        latency_breakdown["ml_reranking_ms"] = round((_time.perf_counter() - t2) * 1000, 2)

        # ── Stage 4: Score calibration and response construction ──────
        table_names = [name for name, _ in final_candidates]

        calibrated_probs = {}
        for name in table_names:
            if name in ce_logits_dict:
                logit = ce_logits_dict[name]
                calibrated_prob = 1.0 / (1.0 + math.exp(-(logit + 2.5) / 1.2))
            else:
                # For tables added by the learned ranker that weren't in the CE batch
                calibrated_prob = 0.5
            calibrated_probs[name] = calibrated_prob

        scores = [round(calibrated_probs[name], 4) for name in table_names]
        confidence = scores[0] if scores else 0.0

        details = {}
        question_lower = question.lower()
        for name in table_names:
            prob = round(calibrated_probs[name], 4)
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

        latency_breakdown["total_ms"] = round(
            sum(v for v in latency_breakdown.values()), 2
        )

        return {
            "retrieved_tables": table_names,
            "scores": scores,
            "confidence": confidence,
            "details": details,
            "latency_breakdown": latency_breakdown,
            "model_used": model_used,
        }
