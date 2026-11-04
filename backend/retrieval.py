from embeddings import generate_query_embedding
from vector_store import get_collection


def retrieve_chunks(question: str, top_k: int = 3):
    """
    Retrieve the most relevant policy chunks
    for an employee question.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    collection = get_collection()

    total_chunks = collection.count()

    if total_chunks == 0:
        return []

    # 1. Generate question embedding
    query_embedding = generate_query_embedding(question)

    # 2. Search ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, total_chunks),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    # 3. Extract retrieved results
    retrieved_chunks = []

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, document, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances
    ):

        retrieved_chunks.append({
            "id": chunk_id,
            "text": document,
            "metadata": metadata,
            "distance": float(distance)
        })

    return retrieved_chunks