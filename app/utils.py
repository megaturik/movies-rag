import chromadb
from schemas import Chunk, SearchRequest, SearchResponse
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def chromadb_search(
        search_request: SearchRequest,
        chroma_client: chromadb.HttpClient,
        collection_name
):
    collection = chroma_client.get_collection(collection_name)
    embeddings = model.encode(search_request.query).tolist()
    search_results = collection.query(
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
