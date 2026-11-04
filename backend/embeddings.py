import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-embedding-001"


@lru_cache(maxsize=1)
def get_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("GEMINI_API_KEY is missing.")

    return genai.Client(api_key=api_key)


def generate_embeddings(texts: list[str]) -> list[list[float]]:

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Empty text cannot be embedded.")

    client = get_client()

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        )
    )

    if not response.embeddings or len(response.embeddings) != len(texts):
        raise RuntimeError("Embedding response count mismatch.")

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def generate_query_embedding(query: str) -> list[float]:

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    client = get_client()

    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=[query],
        config=types.EmbedContentConfig(
            task_type="QUESTION_ANSWERING"
        )
    )

    if not response.embeddings:
        raise RuntimeError("Failed to generate query embedding.")

    return response.embeddings[0].values