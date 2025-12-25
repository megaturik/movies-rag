from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from app.redis_cache import get_redis_cache, set_redis_cache
from app.schemas import AgentResponse, SearchRequest, SearchResponse
from app.settings import get_settings
from app.utils import chromadb_search, get_xai_response

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = Redis(host='localhost', port=6379)
    app.state.chroma_client = await chromadb.AsyncHttpClient(
        host=settings.CHROMADB_HOST,
        port=settings.CHROMADB_PORT
    )
    yield

app = FastAPI(lifespan=lifespan)


@app.exception_handler(Exception)
def exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


@app.post('/api/v1/movies/vector', response_model=SearchResponse)
async def search(
    request: Request,
    search_request: SearchRequest,
):
    cached = await get_redis_cache(request)
    if cached:
        return Response(content=cached)
    data = await chromadb_search(request, search_request, 'movies')
    await set_redis_cache(request, data)
    return data


@app.post('/api/v1/movies/agent', response_model=AgentResponse)
async def agent_ask(request: Request, agent_request: SearchRequest):
    cached = await get_redis_cache(request)
    if cached:
        return Response(content=cached)
    prompt_parts = []
    system_message = (
        "Ты — помощник. Используй ТОЛЬКО информацию из контекста ниже."
        "Если ответа в контексте нет — скажи, что не знаешь."
    )
    search_response = await chromadb_search(request, agent_request, 'movies')
    for chunk in search_response.results:
        prompt_parts.append(
            f"Название фильма: {chunk.metadata['doc_name']}\n"
            f"Режиссер: {chunk.metadata['doc_director']}\n"
            f"Год: {chunk.metadata['doc_year']}\n"
            f"Актеры: {chunk.metadata['doc_actors']}\n"
            f"Сюжет: {chunk.text}"
        )
    context = "\n\n---\n\n".join(prompt_parts)
    prompt = f"""
    Контекст:
    {context}
    Вопрос:
    {agent_request.query}
    """
    data = await get_xai_response(system_message, prompt)
    await set_redis_cache(request, data)
    return data
