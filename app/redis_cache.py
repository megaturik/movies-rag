import hashlib
import json
from typing import Union

from fastapi.requests import Request

from app.schemas import AgentResponse, SearchResponse
from app.settings import get_settings

settings = get_settings()


def build_cache_key(request: Request, body: bytes) -> str:
    payload = {
        "query": dict(request.query_params),
        "body": body.decode()
    }

    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"cache:{request.url.path}:{digest}"


async def cache_key_middleware(request: Request, call_next):
    body = await request.body()
    request.state.cache_key = build_cache_key(request, body)
    return await call_next(request)


async def get_redis_cache(request: Request) -> Union[
    AgentResponse,
    SearchResponse
]:
    redis_client = request.app.state.redis_client
    cache_key = request.state.cache_key
    data = await redis_client.get(cache_key)
    if data:
        return data.decode("utf-8")


async def set_redis_cache(
    request: Request,
    data: Union[AgentResponse, SearchResponse]
):
    redis_client = request.app.state.redis_client
    cache_key = request.state.cache_key
    json_data = data.model_dump_json()
    await redis_client.set(cache_key, json_data, ex=settings.REDIS_CACHE_TTL)
