import json
import logging
from pathlib import Path

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

MOVIES_PATH = Path('./json-data/movies')
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_SIZE = 384
CHROMA_DB_HOST = 'localhost'
CHROMA_DB_PORT = 8010

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


def get_chroma_client():
    client = chromadb.HttpClient(
        host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
    return client


def get_data_from_json_file(file: str) -> tuple[dict, str]:
    excepted_keys = (
        'name', 'year', 'runtime',
        'actors', 'director', 'storyline',
    )
    with open(file) as f:
        data = json.load(f)
    if any(key not in data for key in excepted_keys):
        raise ValueError(f'All of {excepted_keys} should be in data')
    doc_fname = Path(file).name
    doc_mtime = Path(file).stat().st_mtime
    doc_year = data['year']
    doc_actors = ", ".join(data['actors'] if isinstance(
        data['actors'], list) else data['actors'])
    doc_director = ", ".join(data['director']) if isinstance(
        data['director'], list) else data['director']
    metadata = {
        'doc_name': data['name'],
        'doc_year': data['year'],
        'doc_actors': doc_actors,
        'doc_director': doc_director,
        'doc_fname': doc_fname,
        'doc_mtime': doc_mtime,
        'doc_uniq_key': f"{doc_fname}_{doc_year}_{doc_mtime}"
    }
    data = data['storyline']
    return metadata, data


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", ". ", "! ", "? ", " "]
    )
    chunks = splitter.split_text(text)

    return chunks


def embed_chunks(chunks: list) -> list[list[float]]:
    embeddings = model.encode(
        chunks, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def add_to_chroma(
    metadata: dict, chunks: list,
    embeddings: list[list[float]],
    collection
):
    doc_uniq_key = metadata['doc_uniq_key']
    doc_fname = metadata['doc_fname']
    doc_exists = collection.get(where={'doc_uniq_key': doc_uniq_key})
    if doc_exists['documents'][0]:
        logger.info(
            f'Skipping: {doc_fname} with same metadata already exists')
        return
    collection.delete(where={'doc_uniq_key': metadata['doc_uniq_key']})
    logger.info(
        f'Adding/Updating: {doc_fname}')
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{metadata['doc_uniq_key']}_chunk_{i}" for i in range(
            len(chunks))],
        metadatas=[metadata] * len(chunks)
    )


def main():
    client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
    collection = client.get_or_create_collection(name="movies")

    for file in MOVIES_PATH.rglob('*.json'):
        try:
            metadata, data = get_data_from_json_file(file)
            chunks = chunk_text(data)
            embeddings = embed_chunks(chunks)
            add_to_chroma(metadata, chunks, embeddings, collection)
        except Exception as e:
            logger.error(f'Error {e} while proccessing {file}')


if __name__ == '__main__':
    main()
