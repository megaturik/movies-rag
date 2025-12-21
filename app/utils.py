import chromadb
from openai import OpenAI
from schemas import AgentResponse, Chunk, SearchRequest, SearchResponse
from sentence_transformers import SentenceTransformer
from settings import get_settings

settings = get_settings()

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def chromadb_search(
        search_request: SearchRequest,
        chroma_client: chromadb.HttpClient,
        collection_name
) -> SearchResponse:
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


def get_xai_response(
    system_message: str,
    prompt: str,
) -> AgentResponse:
    client = OpenAI(api_key=settings.XAI_API_KEY,
                    base_url=settings.XAI_API_URL)
    response = client.chat.completions.create(
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
