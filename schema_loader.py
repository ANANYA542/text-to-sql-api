from __future__ import annotations

import os
from pathlib import Path
import logging

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_ROOT / "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_beaver_schema(split: str = "dw") -> dict[str, str]:
    """
    Loads the Beaver dataset from HuggingFace and builds enriched
    table descriptions for each table in the schema.

    Returns a dict like:
        {"table_name": "enriched description string", ...}
    """
    logger.info(f"Loading beaver-table dataset, split={split}")
    try:
        table_ds = load_dataset("beaverbench/beaver-table", split=split)
    except Exception as e:
        logger.error(f"Failed to load beaver-table: {e}")
        raise RuntimeError(f"Could not load beaver-table dataset: {e}")

    schema = {}
    for row in table_ds:
        table_name = row["table_name"]
        column_names_raw = row.get("column_names", "")
        column_types_raw = row.get("column_types", "")

        import json
        try:
            col_names = json.loads(column_names_raw)
        except Exception:
            col_names = [c.strip(" []\"'") for c in column_names_raw.split(",") if c.strip()]

        try:
            col_types = json.loads(column_types_raw)
        except Exception:
            col_types = [c.strip(" []\"'") for c in column_types_raw.split(",") if c.strip()]

        # build a nice description for each column
        col_parts = []
        for i, name in enumerate(col_names):
            dtype = col_types[i] if i < len(col_types) else "unknown"
            col_parts.append(f"{name} ({dtype})")

        columns_str = ", ".join(col_parts) if col_parts else column_names_raw

        # Define expansion for common abbreviations in Beaver schema
        abbrev_map = {
            "FCLT": "Facilities, building, room, space, location",
            "SIS": "Student Information System, academic, course, class, enrollment, department",
            "TIP": "Textbook information, course material, textbook, book",
            "HR": "Human Resources, employee, organization, department, job",
            "ZPM": "Facilities rooms load, space, room, building",
            "FAC": "Facilities, building, room, space",
            "ORG": "Organization, department, organizational unit",
            "DLC": "Department, Lab, or Center, organizational DLC",
            "IAP": "Independent Activities Period, January term, activities, academic term",
            "GP": "Grade Point, grading basis, grades",
            "ACAD": "Academic, school, term, semester",
            "DEPT": "Department, academic department",
            "HIST": "History, historical records, archive, temporal",
            "SMRY": "Summary, aggregation",
        }

        expanded_terms = []
        for part in table_name.split("_"):
            expanded_terms.append(part.lower())
            if part in abbrev_map:
                expanded_terms.append(abbrev_map[part].lower())

        concept_context = ", ".join(expanded_terms)

        # enriched description that gives embedding models more context
        enriched = (
            f"{table_name} table. "
            f"Columns: {columns_str}. "
            f"This table can be used in queries involving: {concept_context}."
        )

        schema[table_name] = enriched

    logger.info(f"Loaded {len(schema)} tables from beaver-table/{split}")
    return schema


def load_beaver_queries(split: str = "dw"):
    """
    Loads the query dataset. Each row has a question, gold SQL, and
    the list of tables needed.

    Returns the raw HuggingFace dataset object.
    """
    logger.info(f"Loading beaver-query dataset, split={split}")
    try:
        query_ds = load_dataset("beaverbench/beaver-query", split=split)
    except Exception as e:
        logger.error(f"Failed to load beaver-query: {e}")
        raise RuntimeError(f"Could not load beaver-query dataset: {e}")

    logger.info(f"Loaded {len(query_ds)} queries from beaver-query/{split}")
    return query_ds
