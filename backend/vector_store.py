from pathlib import Path
from functools import lru_cache

import chromadb


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

COLLECTION_NAME = "hr_policy_chunks"


@lru_cache(maxsize=1)
def get_collection():
    """
    Initialize persistent ChromaDB and get the collection.
    """

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    return collection


def store_chunks(chunks, embeddings):
    """
    Store document chunks, embeddings and metadata.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    if not chunks:
        return 0

    collection = get_collection()

    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],

        documents=[
            chunk["text"] for chunk in chunks
        ],

        embeddings=embeddings,

        metadatas=[
            chunk["metadata"] for chunk in chunks
        ]
    )

    return len(chunks)