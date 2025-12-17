import chromadb
from settings import get_settings

settings = get_settings()


def get_chroma_client():
    client = chromadb.HttpClient(
        host=settings.CHROMADB_HOST, port=settings.CHROMADB_PORT)
    return client
