import json
import hashlib
from fastapi.requests import Request
from typing import Union
from app.schemas import SearchResponse, AgentResponse


async def get_body_from_chunks(request: Request) -> bytes:
    data = b''
    async for chunk in request.stream():
        data += chunk
    return data


def build_cache_key(request: Request, body: bytes) -> str:
    payload = {
        "path": request.url.path,
        "query": dict(request.query_params),
        "body": body.decode()
    }

    raw = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"cache:{digest}"


async def get_redis_cache(request: Request) -> Union[
    AgentResponse,
    SearchResponse
]:
    redis_client = request.app.state.redis_client
    body = await get_body_from_chunks(request)
    cache_key = build_cache_key(request, body)
    data = await redis_client.get(cache_key)
    if data:
        return data.decode("utf-8")


async def set_redis_cache(
    request: Request,
    data: Union[AgentResponse, SearchResponse]
):
    redis_client = request.app.state.redis_client
    body = await get_body_from_chunks(request)
    cache_key = build_cache_key(request, body)
    json_data = data.model_dump_json()
    await redis_client.set(cache_key, json_data, ex=300)
