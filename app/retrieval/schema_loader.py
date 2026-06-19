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
    "TIP": "textbook details, reading materials, course reserve lists, and enrolled student counts",
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
    "SMRY": "summary and aggregated metrics across subjects, courses, and enrollment",
    "CIS": "course catalog, course information, subjects offered, and degree details",
    "MATRL": "materials, reserve materials, and textbook detail records",
    "CATALOG": "course catalog entries, subjects offered by term and academic year",
    "LIBRARY": "library course reserves, subjects offered with library materials, and enrolled student counts",
}

# Table-specific usecase overrides for tables with weak or ambiguous auto-generated descriptions.
# These are keyed by TABLE_NAME (uppercase) and completely replace the auto-generated usecase sentence.
TABLE_SPECIFIC_USECASES: dict[str, str] = {
    # COURSE_CATALOG tables: canonical source for subjects/courses offered by academic year and term
    "COURSE_CATALOG_SUBJECT_OFFERED": (
        "This table is the canonical source for subjects offered in a specific academic year and term. "
        "Use this table for queries about: which courses are offered in 2022 or another academic year, "
        "total units per course by year, subjects offered by department per term, "
        "degree-granting subjects, average/variance of total units per academic year."
    ),
    "CIS_COURSE_CATALOG": (
        "This table is the official course catalog listing all MIT courses by course number, "
        "department number, and academic description. "
        "Use specifically for queries about: unique course listings by course number, "
        "courses per school or department count, or catalog-level course data. "
        "Do NOT use for term-by-term subject offering queries — use COURSE_CATALOG_SUBJECT_OFFERED instead."
    ),
    # ACADEMIC TERMS: distinguish _ALL (all historical terms) from ACADEMIC_TERMS (current active terms)
    "ACADEMIC_TERMS_ALL": (
        "This table contains ALL academic terms ever recorded, including historical and current terms, "
        "with term codes, descriptions, academic year, and term status indicators. "
        "Use for queries referencing: academic year descriptions, all terms across years, "
        "terms with a specific term status indicator (e.g., P), or historical semester data."
    ),
    "ACADEMIC_TERMS": (
        "This table contains current and active academic terms with term codes, descriptions, "
        "term start dates, term end dates, and academic year. "
        "Use for queries that filter by: term start date, regular terms (not historical), "
        "fall term, spring term, summer term, or terms starting after a specific date. "
        "This is the standard terms table for most term-date filtering queries."
    ),
    "ACADEMIC_TERM_PARAMETER": (
        "This table stores academic term parameters including current term flags and term selector values. "
        "Use for queries that filter subjects offered in current terms or need term parameter constraints."
    ),
    # FCLT (Facilities) vs FAC: FCLT is the primary facilities data warehouse source
    "FCLT_BUILDING": (
        "This is the primary facilities data warehouse table for MIT buildings. "
        "Contains building names, building keys, building numbers, and building attributes. "
        "Use for queries about buildings, building counts, building names, or top buildings by room count."
    ),
    "FCLT_ROOMS": (
        "This is the primary facilities data warehouse table for MIT rooms. "
        "Contains room details including area, access level, organization, department assignments, "
        "and room parameters. Use for queries about rooms, room area, room access level, "
        "or organization-to-room assignments."
    ),
    # SUBJECT_OFFERED tables: clarify which is the enrollment/stats summary
    "SUBJECT_OFFERED_SUMMARY": (
        "This table provides per-subject aggregated statistics including total units, enrolled student counts, "
        "course number, and subject title. "
        "Use for queries computing: coefficient of variation (CoV), standard deviation (STDDEV), "
        "variance, average, minimum, or maximum of total units or enrolled students grouped by department or subject. "
        "This is the primary table for department-level total-units statistical analysis."
    ),
    "TIP_SUBJECT_OFFERED": (
        "This table links textbook/reading material reserves to subjects offered per term. "
        "It contains enrolled student counts per subject per term. "
        "Use for queries about: enrollment counts per subject, which subjects have library reserve materials, "
        "or subjects offered by department with enrollment statistics."
    ),
    "LIBRARY_SUBJECT_OFFERED": (
        "This table links library subjects offered to course reserves. "
        "Contains library subject offering keys linked to library reserve catalogs and course materials. "
        "Use for queries about which subjects have library reserves or library course material associations."
    ),
    "LIBRARY_RESERVE_MATRL_DETAIL": (
        "This table stores detailed library reserve material records per subject offering and term. "
        "It contains material status codes (e.g., 'U' for unavailable) for each reserve item. "
        "Use for queries about: library reserve materials per course, material status filtering, "
        "subjects with at least one library reserve material, or reserve material detail by term."
    ),

    # SUBJECT_SUMMARY: per-term enrollment summary (distinct from SUBJECT_OFFERED_SUMMARY)
    "SUBJECT_SUMMARY": (
        "This table stores per-term summary enrollment data for subjects. "
        "It contains enrolled student counts aggregated per subject per term. "
        "Use for queries about: total enrollment per subject per term, "
        "average enrollment per subject, top subjects by enrollment count, "
        "or spring/fall term enrollment statistics per subject."
    ),

    # TIP_DETAIL: raw textbook/ISBN-level detail records
    "TIP_DETAIL": (
        "This table stores individual textbook and reading material detail records including "
        "ISBN, author, title, edition, publisher, and material type. "
        "Use for queries about: specific textbook ISBNs, authors, titles, editions, publishers, "
        "or book-level details linked to courses."
    ),

    # STUDENT_DEPARTMENT: maps student departments metadata
    "STUDENT_DEPARTMENT": (
        "This table stores academic department metadata (like school and department code). "
        "It does NOT contain individual student records."
    ),

    # SIS_ADMIN_DEPARTMENT: administrative department info (school name, phone, degree-granting flag)
    "SIS_ADMIN_DEPARTMENT": (
        "This table stores administrative information about academic departments, including "
        "school name, department phone number, degree-granting status, department codes, "
        "and financial data such as clearing cost collector values. "
        "Use for queries that need: school name per department, department phone number, "
        "whether a department is degree-granting, clearing cost collector for departments, "
        "or the school a department belongs to."
    ),

    # SIS_SUBJECT_CODE: subject/course code lookup table
    "SIS_SUBJECT_CODE": (
        "This table is the master lookup for subject and course codes, including subject ID, "
        "subject number, department, and course identifiers. "
        "Use for queries that need course codes, subject numbers, or subject-to-department mappings."
    ),

    # SIS_COURSE_DESCRIPTION: course description and metadata
    "SIS_COURSE_DESCRIPTION": (
        "This table stores detailed course descriptions, course titles, and course metadata "
        "such as units and grading basis. "
        "Use for queries about course descriptions, course titles, units, or letter-graded courses."
    ),

    # IAP_SUBJECT_DETAIL: Independent Activities Period subject details
    "IAP_SUBJECT_DETAIL": (
        "This table stores details about Independent Activities Period (IAP/January term) subjects, "
        "including IAP course titles, type, dates, and instructors. "
        "Use for queries about IAP subjects, January term courses, or IAP course details."
    ),

    # LIBRARY_COURSE_INSTRUCTOR: links library course reserves to instructors
    "LIBRARY_COURSE_INSTRUCTOR": (
        "This table links library course reserve records to course instructors. "
        "Use for queries about which instructor teaches a course that has library reserves, "
        "or instructor-course-library reserve associations."
    ),

    # DRUPAL_COURSE_CATALOG: web-facing course catalog (less authoritative than CIS_COURSE_CATALOG)
    "DRUPAL_COURSE_CATALOG": (
        "This table stores the web-facing Drupal course catalog, a public-facing version of course listings. "
        "Use as a secondary source for course catalog queries when CIS_COURSE_CATALOG is the primary source."
    ),

    # FCLT_ORGANIZATION: MIT facilities organization records (DLC units, departments, organizations)
    "FCLT_ORGANIZATION": (
        "This table stores MIT facilities organization records, including organization numbers, "
        "DLC (Department, Lab, Center) names, organization names, and organization hierarchies. "
        "Use for queries about: organizations associated with DLCs, organization numbers, "
        "DLC-to-organization mappings, or history department DLCs."
    ),

    # FCLT_ORG_DLC_KEY: maps organizations to DLC (Department, Lab, Center) keys
    "FCLT_ORG_DLC_KEY": (
        "This table maps MIT facilities organizations to their DLC (Department, Lab, Center) keys. "
        "Use for queries that join DLCs to organizations, or need organization-DLC relationships "
        "such as 'for each DLC, list the organizations'."
    ),

    # MIT_STUDENT_DIRECTORY: student records including student year, department, degree status
    "MIT_STUDENT_DIRECTORY": (
        "This table is the MIT student directory containing records of all students with fields like "
        "full name, department, student year (G=Graduate, U=Undergraduate), degree type, "
        "and enrollment status. Use COUNT(FULL_NAME) or COUNT(EMAIL_ADDRESS) to count students, as there is no STUDENT_ID column. "
        "Join with SIS_DEPARTMENT on MIT_STUDENT_DIRECTORY.DEPARTMENT = SIS_DEPARTMENT.DEPARTMENT_CODE. "
        "Use for queries about: graduate students (student year = G), undergraduate students, "
        "students by department, student counts, or unique student counts."
    ),

    # SUBJECT_OFFERED: individual subject offerings per term (not summary)
    "SUBJECT_OFFERED": (
        "This table stores individual subject offerings per term, including subject title, "
        "department, total units, term code, and subject identifiers. "
        "Use for queries about: specific subjects offered, total units per subject, "
        "subjects offered by a department per term, or excluding specific departments from subject listings."
    ),
}

def get_table_usecase(table_name: str) -> str:
    """Generates a sentence explaining what a table is useful for based on its abbreviations."""
    # Check table-specific override first (highest priority)
    override = TABLE_SPECIFIC_USECASES.get(table_name.upper())
    if override:
        return override

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
