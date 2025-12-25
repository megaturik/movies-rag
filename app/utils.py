import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from fastapi import Request
from openai import AsyncOpenAI

from app.schemas import AgentResponse, Chunk, SearchRequest, SearchResponse
from app.settings import get_settings

settings = get_settings()

executor = ThreadPoolExecutor()


@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.SENTENCE_MODEL)


async def chromadb_search(
        request: Request,
        search_request: SearchRequest,
        collection_name
) -> SearchResponse:
    model = get_model()
    chroma_client = request.app.state.chroma_client
    collection = await chroma_client.get_collection(collection_name)
    loop = asyncio.get_event_loop()
    embeddings = await loop.run_in_executor(
        executor,
        model.encode,
        search_request.query
    )
    search_results = await collection.query(
        query_embeddings=[embeddings],
        n_results=search_request.top_k,
        include=['documents', 'metadatas']
    )
    documents = search_results.get('documents', [[]])[0]
    metadatas = search_results.get('metadatas', [[]])[0]
    chunks = [
        Chunk.model_validate(
            {"text": doc, "metadata": meta}
        ) for doc, meta in zip(
            documents, metadatas
        )

    ]
    return SearchResponse(results=chunks)


async def get_xai_response(
    system_message: str,
    prompt: str,
) -> AgentResponse:
    client = AsyncOpenAI(
        api_key=settings.XAI_API_KEY,
        base_url=settings.XAI_API_URL)
    response = await client.chat.completions.create(
        model=settings.XAI_MODEL,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=settings.XAI_TEMP,
        max_tokens=settings.XAI_MAX_TOKENS
    )

    response = response.choices[0].message.content
    return AgentResponse(answer=response)
