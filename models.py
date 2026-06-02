from pydantic import BaseModel, Field


class RetrieveRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The natural language question to retrieve tables for",
    )


class TableDetail(BaseModel):
    relevance_score: float
    reason: str


class RetrieveResponse(BaseModel):
    retrieved_tables: list[str]
    scores: list[float]
    confidence: float
    details: dict[str, TableDetail]


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
