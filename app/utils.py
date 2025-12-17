import chromadb
from schema import SearchRequest
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def chromadb_search(
        search_request: SearchRequest,
        chroma_client: chromadb.HttpClient,
        collection_name
):
    collection = chroma_client.get_collection(collection_name)
    embeddings = model.encode(search_request.query).tolist()
    results = collection.query(
        query_embeddings=[embeddings],
        n_results=search_request.top_k,
        include=['documents', 'metadatas', 'distances']
    )
    return results
