"""Turns passages into vectors, stores them, reads a section back and searches them.

This is the only module that talks to the vector store.
"""
import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import config

_BATCH_SIZE = 128
_SEARCH_EFFORT = 2000
_embedding_functions = {}


def _embedding_function():
    if "model" not in _embedding_functions:
        function = SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_NAME,
                                                        device=config.EMBEDDING_DEVICE)
        function(["load check"])
        _embedding_functions["model"] = function
    return _embedding_functions["model"]


def _store_file(store_path):
    return os.path.join(store_path, config.VECTOR_STORE_FILE_NAME)


def build_vector_index(chunk_records, store_path):
    """Build the persistent store afresh from the passages; returns how many it holds."""
    if os.path.isfile(store_path):
        raise ValueError(f"The vector store path {store_path} is a file, not a folder")
    if (os.path.isdir(store_path) and os.listdir(store_path)
            and not os.path.isfile(_store_file(store_path))):
        raise ValueError(f"The folder {store_path} holds files but no "
                         f"{config.VECTOR_STORE_FILE_NAME}, so it is not a vector store")
    embedding = _embedding_function()
    os.makedirs(store_path, exist_ok=True)
    client = chromadb.PersistentClient(path=store_path)
    if config.VECTOR_COLLECTION_NAME in [collection.name for collection in client.list_collections()]:
        client.delete_collection(config.VECTOR_COLLECTION_NAME)
    collection = client.create_collection(
        name=config.VECTOR_COLLECTION_NAME, embedding_function=embedding,
        metadata={"hnsw:space": config.VECTOR_DISTANCE_METRIC,
                  "hnsw:construction_ef": _SEARCH_EFFORT, "hnsw:search_ef": _SEARCH_EFFORT,
                  "hnsw:M": 64})
    for start in range(0, len(chunk_records), _BATCH_SIZE):
        batch = chunk_records[start:start + _BATCH_SIZE]
        collection.add(
            ids=[record["chunk_id"] for record in batch],
            documents=[record["chunk_text"] for record in batch],
            metadatas=[{"document_name": record["document_name"],
                        "section_number": int(record["section_number"]),
                        "passage_number": int(record["passage_number"]),
                        "carried_characters": int(record["carried_characters"])}
                       for record in batch])
    return collection.count()


def open_vector_index(store_path):
    """Open the store for reading and searching, without changing it."""
    if not os.path.isfile(_store_file(store_path)):
        raise ValueError(f"The folder {store_path} holds no {config.VECTOR_STORE_FILE_NAME}")
    client = chromadb.PersistentClient(path=store_path)
    if config.VECTOR_COLLECTION_NAME not in [collection.name
                                             for collection in client.list_collections()]:
        raise ValueError(f"The vector store at {store_path} holds no collection "
                         f"{config.VECTOR_COLLECTION_NAME}")
    return client.get_collection(name=config.VECTOR_COLLECTION_NAME,
                                 embedding_function=_embedding_function())


def read_document_section(collection, document_name, section_number):
    """Read one section of one document back, whole, from its own passages alone."""
    stored = collection.get(
        where={"$and": [{"document_name": document_name},
                        {"section_number": int(section_number)}]},
        include=["documents", "metadatas"])
    if not stored["ids"]:
        raise ValueError(config.NO_SECTION_PASSAGES_MESSAGE.format(
            section_number=section_number, document_name=document_name))
    passages = sorted(zip(stored["metadatas"], stored["documents"]),
                      key=lambda pair: pair[0]["passage_number"])
    text = passages[0][1]
    for metadata, passage in passages[1:]:
        text = text + " " + passage[int(metadata.get("carried_characters", 0)):]
    return text


def search_documents(collection, query_texts):
    """Ask several questions at once; each gets the nearest passages outside the excluded ones."""
    total = collection.count()
    if total == 0:
        return [[] for _ in query_texts]
    results = collection.query(query_texts=list(query_texts), n_results=total,
                               include=["documents", "metadatas", "distances"])
    answers = []
    for ids, documents, metadatas, distances in zip(results["ids"], results["documents"],
                                                    results["metadatas"], results["distances"]):
        passages = [{"chunk_id": chunk_id, "chunk_text": document,
                     "document_name": metadata["document_name"], "distance": float(distance)}
                    for chunk_id, document, metadata, distance
                    in zip(ids, documents, metadatas, distances)
                    if metadata["document_name"] not in config.CHAT_EXCLUDED_DOCUMENTS]
        passages.sort(key=lambda passage: passage["distance"])
        answers.append(passages[:config.CHAT_RETRIEVAL_RESULT_COUNT])
    return answers
