from pydantic import BaseModel, Field

class RetrieveRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The natural language question to retrieve tables for",
    )
    top_k: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of top tables to retrieve"
    )

class GenerateSQLRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The natural language question to generate SQL for",
    )
    use_retrieved_context: bool = Field(
        True,
        description="Whether to use retrieval engine context"
    )

class ExecuteSQLRequest(BaseModel):
    sql: str = Field(..., description="The SQL query to execute")

class ExplainSQLRequest(BaseModel):
    sql: str = Field(..., description="The SQL query to explain")

