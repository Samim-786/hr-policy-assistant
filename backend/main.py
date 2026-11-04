
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, HTTPException

from ingestion import extract_text

from chunking import chunk_document
from embeddings import generate_embeddings
from vector_store import store_chunks, get_collection
from retrieval import retrieve_chunks
from pydantic import BaseModel, Field
from rag import answer_question

app = FastAPI(
    title="HR Policy Assistant",
    description="RAG-based HR policy assistant",
    version="1.0.0"
)

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)



@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "message": "HR Policy Assistant API is running"
    }


@app.post("/documents/upload")
async def upload_policy(file: UploadFile = File(...)):

    # Get a safe filename
    original_name = (file.filename or "").replace("\\", "/").split("/")[-1]

    if not original_name:
        raise HTTPException(
            status_code=400,
            detail="Filename is required."
        )

    # Validate extension
    if Path(original_name).suffix.lower() not in {".md", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only .md and .txt files are supported."
        )

    content = await file.read()
    await file.close()

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    # Unique filename avoids overwriting existing documents
    stored_name = f"{uuid4().hex[:8]}_{original_name}"
    file_path = UPLOAD_DIR / stored_name

    try:
        # Save original document
        file_path.write_bytes(content)

        # Extract document text
        extracted_text = extract_text(file_path)

    except ValueError as exc:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))

    except OSError:
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Unable to save or read the uploaded document."
        )

    return {
        "status": "success",
        "message": "Policy uploaded and text extracted successfully.",
        "filename": original_name,
        "stored_as": stored_name,
        "characters": len(extracted_text)
    }

@app.get("/documents/{stored_name}/chunks")
def get_document_chunks(stored_name: str):

    # Prevent path traversal
    if Path(stored_name).name != stored_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    file_path = UPLOAD_DIR / stored_name

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        text = extract_text(file_path)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    # Remove the generated UUID prefix
    original_name = stored_name[9:]

    chunks = chunk_document(
        text=text,
        document_name=original_name,
        stored_name=stored_name
    )

    return {
        "document": original_name,
        "total_chunks": len(chunks),
        "chunks": chunks
    }

@app.post("/documents/{stored_name}/index")
def index_document(stored_name: str):

    # Validate filename
    if Path(stored_name).name != stored_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename."
        )

    file_path = UPLOAD_DIR / stored_name

    if not file_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    try:
        # 1. Extract document text
        text = extract_text(file_path)

        original_name = stored_name[9:]

        # 2. Generate chunks
        chunks = chunk_document(
            text=text,
            document_name=original_name,
            stored_name=stored_name
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content found in document."
            )

        # 3. Generate embeddings
        texts = [
            chunk["text"] for chunk in chunks
        ]

        embeddings = generate_embeddings(texts)

        # 4. Store everything in ChromaDB
        indexed_count = store_chunks(
            chunks,
            embeddings
        )

        return {
            "status": "success",
            "document": original_name,
            "chunks_indexed": indexed_count,
            "total_chunks_in_database": get_collection().count()
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Document indexing failed. Check server logs."
        )


@app.get("/vector-store/stats")
def vector_store_stats():

    collection = get_collection()

    return {
        "collection": "hr_policy_chunks",
        "total_chunks": collection.count()
    }

class RetrievalRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)

@app.post("/retrieve")
def retrieve_policy_chunks(request: RetrievalRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        chunks = retrieve_chunks(
            question=request.question,
            top_k=request.top_k
        )

        return {
            "question": request.question,
            "total_retrieved": len(chunks),
            "retrieved_chunks": chunks
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Retrieval failed. Check server logs."
        )

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1)

@app.post("/ask")
def ask_question_api(request: QuestionRequest):

    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        result = answer_question(request.question)

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc)
        )

    except Exception as exc:

        print(f"RAG error: {exc}")

        raise HTTPException(
            status_code=502,
            detail="Unable to generate an answer. Please try again."
        )