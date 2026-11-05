# HR Policy Assistant

An AI-powered HR Policy Assistant built using Retrieval-Augmented Generation (RAG).

The application allows employees to ask questions about company policies in natural language. It retrieves relevant information from uploaded documents and generates grounded answers with citations.

The system is designed to avoid unsupported policy answers by verifying supporting evidence and refusing questions when sufficient information is unavailable.

## Live Demo

| Component | URL |
|---|---|
| Streamlit Frontend | https://hr-policy-assistant-hkbcuu6muelqtowaip5hfx.streamlit.app/ |
| FastAPI Backend | https://hr-policy-assistant-yhbp.onrender.com |
| API Health Check | https://hr-policy-assistant-yhbp.onrender.com/health |
| API Documentation | https://hr-policy-assistant-yhbp.onrender.com/docs |

### Using the Deployed Application

1. Open the Streamlit frontend.
2. Navigate to **Admin - Upload Policies**.
3. Upload the three Markdown policy files provided with the assignment:
   - `leave-policy.md`
   - `benefits-policy.md`
   - `it-security-policy.md`
4. Upload and index each document.
5. Verify that the ChromaDB collection `hr_policy_chunks` contains indexed chunks.
6. Navigate to **Employee - Ask Questions**.
7. Ask questions about the uploaded policies and review the answers and citations.

### Deployment and Storage Note

The FastAPI backend is hosted on Render's free web service. Its filesystem is ephemeral, so uploaded documents and the local ChromaDB index can be lost when the service spins down, restarts, or redeploys.

If the collection is empty, upload and index the policy documents again before asking questions.

This is a prototype deployment limitation. A persistent disk or external vector database would be appropriate for long-term storage.

Before testing the application, open the backend health-check endpoint and wait for the Render service to become healthy. The free instance may take some time to wake up after inactivity.

See the [Render free instance limitations](https://render.com/docs/free) for details.

## Features

- Upload Markdown and TXT policy documents.
- Section-aware document chunking.
- Preserve Markdown tables during chunking.
- Generate embeddings using Google's Gemini Embedding API.
- Store document chunks and embeddings in ChromaDB.
- Perform semantic similarity search using cosine distance.
- Generate grounded answers using Groq LLM.
- Structured JSON responses.
- Evidence verification against retrieved document chunks.
- Citations containing document names and section headings.
- Safe refusal for unsupported questions.
- Simple Streamlit interface for administrators and employees.
- Automated evaluation test set.

## Technology Stack

| Component           | Technology                   |
| ------------------- | ---------------------------- |
| Backend             | Python, FastAPI              |
| Frontend            | Streamlit                    |
| Embedding model     | Gemini Embedding 001         |
| Vector database     | ChromaDB                     |
| LLM                 | Groq                         |
| Response validation | JSON Schema and Python       |
| Chunking            | Custom Python implementation |

## Architecture

The application has two main workflows.

### Document Ingestion

Admin uploads a policy document.

```text
Document Upload
      |
      v
Text Extraction
      |
      v
Section-aware Chunking
      |
      v
Gemini Embedding API
      |
      v
ChromaDB
```

### Question Answering

```text
Employee Question
       |
       v
Question Embedding
       |
       v
ChromaDB Retrieval
       |
       v
Relevant Policy Chunks
       |
       v
Groq LLM
       |
       v
Evidence Verification
       |
       v
Citation Generation
       |
       v
Structured JSON Response
```

### AI Models and Providers

- Embeddings: Google Gemini hosted API
  - Model: `gemini-embedding-001`
  - Document task type: `RETRIEVAL_DOCUMENT`
  - Query task type: `QUESTION_ANSWERING`

- LLM: Groq hosted API
  - Default model: `openai/gpt-oss-20b`
  - Model can optionally be overridden using `GROQ_MODEL`.

Both services use hosted API calls; no local LLM or embedding model is required.

### Environment Variables

| Variable | Purpose |
|---|---|
| GEMINI_API_KEY | Gemini embedding API key |
| GROQ_API_KEY | Groq chat API key |
| RAG_DEBUG | Enables debug logging when set to `1` |

## Project Structure

```text
hr-policy-assistant/
│
├── backend/
│   ├── main.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retrieval.py
│   └── rag.py
│
├── frontend/
│   └── app.py
│
│
├── tests/
│   └── evaluate.py
│
├── data/
│   ├── uploads/
│   └── chroma/
│
├── DESIGN.md
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

The `data/` directory is generated locally and stores uploaded documents and the persistent ChromaDB database.

## Requirements

- Python 3.10 or later.
- Gemini API key.
- Groq API key.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Samim-786/hr-policy-assistant.git
cd hr-policy-assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root.

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholders with your API keys.

Never commit your `.env` file to GitHub.

## Running the Application

The application requires two terminal sessions.

### Terminal 1: Start FastAPI

From the project root:

```bash
source .venv/bin/activate

cd backend

uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Terminal 2: Start Streamlit

From the project root:

```bash
source .venv/bin/activate

streamlit run frontend/app.py
```

Frontend:

```text
http://localhost:8501
```

## How to Use

### Administrator

1. Open the Streamlit application.
2. Navigate to the Admin - Upload Policies tab.
3. Select a Markdown or TXT policy document.
4. Upload and index the document.
5. The backend extracts its content, generates chunks and embeddings, and stores them in ChromaDB.

### Employee

1. Navigate to the Employee - Ask Questions tab.
2. Enter a question about company policies.
3. The backend retrieves relevant policy chunks.
4. The LLM generates an answer using the retrieved context.
5. The application displays the answer and supporting citations.

When sufficient information is unavailable, the application returns a safe refusal.

## Evaluation

The project includes an automated evaluation script containing seven test cases.

The evaluation covers:

- Direct factual questions.
- Table-based questions.
- Explicit negative policies.
- Unsupported questions.
- Citation correctness.

Run the evaluation with FastAPI running:

```bash
python tests/evaluate.py
```

In addition to the automated evaluation, the application was manually stress-tested using a separate travel and expense policy containing 20 questions.

These tests covered:

- Numerical calculations.
- Table interpretation.
- Conditional rules.
- Approval requirements.
- Expense restrictions.
- Missing policy information.
- Evidence and citation correctness.

## Example API Response

Request:

```json
{
  "question": "How many casual leave days can I carry forward?"
}
```

Response:

```json
{
  "status": "answered",
  "answer": "A maximum of 8 days of unused casual leave may be carried forward.",
  "citations": [
    {
      "document_name": "leave-policy.md",
      "section": "Leave Policy > 4. Carry-forward and encashment > 4.1 Casual leave carry-forward"
    }
  ]
}
```

For unsupported questions:

```json
{
  "status": "insufficient_information",
  "answer": "I don't have enough information in the uploaded policies. Please contact HR.",
  "citations": []
}
```

## Design Decisions

- Custom Python orchestration instead of LangChain to maintain control over the RAG pipeline.
- Section-aware chunking to preserve policy structure.
- Table-aware chunking to retain relationships between table headings and values.
- Gemini embeddings to avoid maintaining a local machine-learning model.
- ChromaDB for lightweight persistent vector storage.
- Evidence verification and metadata-based citations to improve answer traceability.

See [DESIGN.md](DESIGN.md) for the complete architecture, retrieval strategy, grounding approach and design trade-offs.

## Current Limitations

- Supports Markdown and TXT documents; PDF extraction is not implemented.
- Depends on external Gemini and Groq APIs.
- API availability and usage quotas can affect answer generation.
- The current prototype does not implement production authentication or authorization.
- Evidence verification checks source traceability but does not mathematically guarantee that every generated answer is entailed by its evidence.

## Future Improvements

- Authentication and role-based authorization.
- Hybrid keyword and vector search.
- Policy versioning and re-indexing.
- Improved PDF and complex table extraction.
- Expanded retrieval and answer evaluation.
- Feedback collection and monitoring.

## Security

API keys must be stored in environment variables and excluded from version control.

Review external provider data-handling requirements before processing confidential company policies.

---

**Built as a take-home assignment demonstrating Retrieval-Augmented Generation, semantic retrieval, grounded LLM responses and evidence-based citations.**
