from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from app.core import config
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Map common abbreviations to readable meanings to generate "useful for" sentences
ABBREV_MAP = {
    "FCLT": "facilities, buildings, rooms, spaces, and locations",
    "SIS": "Student Information System, academic records, courses, and department enrollments",
    "TIP": "textbook details, reading materials, and course reserve lists",
    "HR": "human resources, employees, jobs, and organizational hierarchies",
    "ZPM": "facilities room loading, space utilization, and architecture",
    "FAC": "facilities, physical buildings, and room parameters",
    "ORG": "organizational structure, units, and departments",
    "DLC": "Departments, Labs, and Centers (DLC)",
    "IAP": "Independent Activities Period (January term) course offerings",
    "GP": "Grade Point averages and grading basis details",
    "ACAD": "academic timelines, terms, and semesters",
    "DEPT": "departments and schools within the university",
    "HIST": "historical tracking and temporal archive records",
    "SMRY": "summary metrics and aggregated data views",
}

def get_table_usecase(table_name: str) -> str:
    """Generates a sentence explaining what a table is useful for based on its abbreviations."""
    parts = table_name.upper().split("_")
    meanings = []
    for p in parts:
        if p in ABBREV_MAP:
            meanings.append(ABBREV_MAP[p])
    
    if meanings:
        return f"This table is useful for queries concerning {', '.join(meanings)}."
    return f"This table stores records related to {table_name.lower().replace('_', ' ')}."

def is_ignored_col(col_name: str) -> bool:
    """Checks if a column should be ignored completely (metadata/audit columns)."""
    c = col_name.lower()
    return any(x in c for x in ("warehouse_load_date", "last_activity_date", "load_date", "record_status", "load_dt", "update_date"))

def is_join_key_column(col_name: str) -> bool:
    """Checks if a column name is a key/identifier suitable for generating table relationships/joins."""
    c = col_name.lower()
    if is_ignored_col(c):
        return False
    # Check common key suffixes
    if any(c.endswith(suffix) for suffix in ("_id", "_key", "_code", "_number", "_num", "_name")):
        # Exclude generic codes/statuses that are not useful join keys
        if any(x in c for x in ("level_code", "rule_code", "type_code", "status_code", "basis_code", "state_code")):
            return False
        return True
    # Explicitly known join keys
    if c in ("subject_id", "term_code", "department_code", "building_key", "room_number", "student_id", "employee_id", "department", "school", "course", "building", "room"):
        return True
    return False

def load_beaver_schema(split: str = "dw") -> dict[str, str]:
    """
    Loads the Beaver dataset and builds highly enriched table descriptions.
    Enrichment uses column roles (foreign keys, numeric aggregations, text, temporal)
    and cross-table relations based on shared column names.
    """
    logger.info(f"Loading beaver-table dataset, split={split}")
    try:
        table_ds = load_dataset("beaverbench/beaver-table", split=split)
    except Exception as e:
        logger.error(f"Failed to load beaver-table: {e}")
        raise RuntimeError(f"Could not load beaver-table dataset: {e}")

    # First pass: parse all table columns and build indexing map of column -> tables
    table_cols = {}
    col_to_tables = {}
    raw_rows = []

    for row in table_ds:
        table_name = row["table_name"]
        raw_rows.append(row)
        
        column_names_raw = row.get("column_names", "")
        column_types_raw = row.get("column_types", "")

        try:
            col_names = json.loads(column_names_raw)
        except Exception:
            col_names = [c.strip(" []\"'") for c in column_names_raw.split(",") if c.strip()]

        try:
            col_types = json.loads(column_types_raw)
        except Exception:
            col_types = [c.strip(" []\"'") for c in column_types_raw.split(",") if c.strip()]

        # Ensure matching lengths
        if len(col_types) < len(col_names):
            col_types += ["unknown"] * (len(col_names) - len(col_types))

        table_cols[table_name] = list(zip(col_names, col_types))

        for col in col_names:
            col_lower = col.lower()
            if col_lower not in col_to_tables:
                col_to_tables[col_lower] = []
            col_to_tables[col_lower].append(table_name)

    schema = {}
    for row in raw_rows:
        table_name = row["table_name"]
        columns = table_cols[table_name]

        col_descriptions = []
        relations_grouped = {}

        for col_name, col_type in columns:
            col_lower = col_name.lower()
            if is_ignored_col(col_lower):
                continue
            role = None

            # Apply column role rules
            if col_lower.endswith("_id"):
                ref_table = col_lower[:-3]
                role = f"foreign key referencing {ref_table} table"
            elif any(x in col_lower for x in ["count", "num", "total", "sum", "amount"]):
                role = "numeric aggregation column"
            elif any(x in col_lower for x in ["name", "title", "label", "desc", "code"]):
                role = "descriptive text column"
            elif any(x in col_lower for x in ["date", "time", "year", "term", "timestamp"]):
                role = "temporal column"

            col_desc = f"{col_name} ({col_type})"
            if role:
                col_desc += f" [role: {role}]"
            col_descriptions.append(col_desc)

            # Find related tables sharing the same key column
            if is_join_key_column(col_lower):
                sharing = col_to_tables.get(col_lower, [])
                for other_t in sharing:
                    if other_t != table_name:
                        if other_t not in relations_grouped:
                            relations_grouped[other_t] = []
                        relations_grouped[other_t].append(col_name)

        relations_list = []
        for other_t, cols in relations_grouped.items():
            cols_str = "/".join(sorted(list(set(cols))))
            relations_list.append(f"{other_t} (via {cols_str})")

        columns_str = ", ".join(col_descriptions)
        usecase_str = get_table_usecase(table_name)
        
        relations_str = ""
        if relations_list:
            uniq_relations = sorted(relations_list)
            relations_str = f" Related Tables (Join Paths): [{', '.join(uniq_relations)}]."

        # Final enriched description
        enriched = (
            f"Table: {table_name}. "
            f"{usecase_str} "
            f"Schema structure: [{columns_str}]. "
            f"{relations_str}"
        )
        schema[table_name] = enriched

    logger.info(f"Loaded and enriched {len(schema)} tables from beaver-table/{split}")
    return schema

def load_beaver_queries(split: str = "dw"):
    """Loads the query dataset containing natural language questions and ground truth SQL."""
    logger.info(f"Loading beaver-query dataset, split={split}")
    try:
        query_ds = load_dataset("beaverbench/beaver-query", split=split)
    except Exception as e:
        logger.error(f"Failed to load beaver-query: {e}")
        raise RuntimeError(f"Could not load beaver-query dataset: {e}")

    logger.info(f"Loaded {len(query_ds)} queries from beaver-query/{split}")
    return query_ds
