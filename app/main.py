from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.chroma import get_chroma_client
from app.schemas import AgentResponse, SearchRequest, SearchResponse
from app.settings import get_settings
from app.utils import chromadb_search, get_xai_response

settings = get_settings()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


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
    chroma_client=Depends(get_chroma_client)
):
    return await chromadb_search(search_request, chroma_client, 'movies')


@app.post('/api/v1/movies/agent', response_model=AgentResponse)
async def agent_ask(
    request: Request,
    agent_request: SearchRequest,
    chroma_client=Depends(get_chroma_client)
):
    prompt_parts = []
    system_message = (
        "Ты — помощник. Используй ТОЛЬКО информацию из контекста ниже."
        "Если ответа в контексте нет — скажи, что не знаешь."
    )
    search_response = await chromadb_search(
        agent_request,
        chroma_client,
        'movies'
    )
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
    results = await get_xai_response(system_message, prompt)
    return results
