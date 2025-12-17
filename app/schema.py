from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int


class Chunk(BaseModel):
    chunk: str
    metadata: dict
    distance: float


class SearchResponse(BaseModel):
    results: list[Chunk]
