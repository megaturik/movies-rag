from chroma import get_chroma_client
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schema import SearchRequest
from settings import get_settings
from utils import chromadb_search

settings = get_settings()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/api/v1/movies/search')
def search(search_request: SearchRequest,
           chroma_client=Depends(get_chroma_client)):
    return chromadb_search(search_request, chroma_client, 'movies')
