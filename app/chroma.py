import chromadb

from app.settings import get_settings

settings = get_settings()


async def get_chroma_client():
    client = await chromadb.AsyncHttpClient(
        host=settings.CHROMADB_HOST,
        port=settings.CHROMADB_PORT
    )
    return client