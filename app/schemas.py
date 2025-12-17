from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k: int


class Chunk(BaseModel):
    text: str
    metadata: dict


class SearchResponse(BaseModel):
    results: list[Chunk]
