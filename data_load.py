import json
import logging
import os
import sys
from pathlib import Path

import chromadb
import tiktoken
from sentence_transformers import SentenceTransformer

MOVIES_PATH = './json-data/movies'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_SIZE = 384
CHROMA_DB_HOST = 'localhost'
CHROMA_DB_PORT = 8010

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

enc = tiktoken.get_encoding("cl100k_base")
model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.HttpClient(host=CHROMA_DB_HOST, port=CHROMA_DB_PORT)
collection = client.get_or_create_collection(name="movies")


def get_data_from_json_file(file: str) -> tuple[dict, str]:
    excepted_keys = (
        'name', 'year', 'runtime',
        'actors', 'director', 'storyline',
    )
    with open(file) as f:
        data = json.load(f)
    if any(key not in data for key in excepted_keys):
        raise ValueError(f'All of {excepted_keys} should be in data')
    doc_fname, doc_mtime = Path(file).name, Path(file).stat().st_mtime
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
        'doc_unique_key': f"{doc_fname}_{doc_mtime}"
    }
    data = data['storyline']
    return metadata, data


def chunk_text(text, max_tokens=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list:
    tokens = enc.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(enc.decode(chunk_tokens))
        start = end - overlap

    return chunks


def embed_chunks(chunks: list) -> list[list[float]]:
    embeddings = model.encode(
        chunks, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.tolist()


def add_to_chroma(metadata: dict, chunks: list, embeddings: list[list[float]]):
    doc_uniq_key = metadata['doc_unique_key']
    doc_fname = metadata['doc_fname']
    doc_exists = collection.query(
        query_embeddings=[[0] * EMBEDDING_SIZE],
        n_results=1,
        where={'doc_unique_key': doc_uniq_key}
    )
    if doc_exists['documents'][0]:
        logging.info(
            f'Skipping: {doc_fname} with same metadata already exists')
        return
    collection.delete(where={'doc_fname': metadata['doc_fname']})
    logging.info(
        f'Adding/Updating: {doc_fname}')
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"{metadata['doc_fname']}_chunk_{i}" for i in range(len(chunks))],
        metadatas=[metadata] * len(chunks)
    )


def main():
    for dirpath, dirnames, filenames in os.walk(MOVIES_PATH):
        for filename in filenames:
            if filename.endswith('.json'):
                filepath = os.path.join(dirpath, filename)
                try:
                    metadata, data = get_data_from_json_file(filepath)
                    chunks = chunk_text(data)
                    embeddings = embed_chunks(chunks)
                    add_to_chroma(metadata, chunks, embeddings)
                except Exception as e:
                    logging.info(f'Error {e} while proccessing {filename}')


main()
