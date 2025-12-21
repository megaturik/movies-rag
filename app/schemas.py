from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)


class Chunk(BaseModel):
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: list[Chunk]


class AgentResponse(BaseModel):
    answer: str
