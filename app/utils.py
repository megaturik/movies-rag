import asyncio
from concurrent.futures import ThreadPoolExecutor

import chromadb
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from app.schemas import AgentResponse, Chunk, SearchRequest, SearchResponse
from app.settings import get_settings

settings = get_settings()

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

executor = ThreadPoolExecutor()


async def chromadb_search(
        search_request: SearchRequest,
        chroma_client: chromadb.AsyncHttpClient,
        collection_name
) -> SearchResponse:
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
        temperature=0.7,
        max_tokens=3000
    )

    response = response.choices[0].message.content
    return AgentResponse(answer=response)
