# HR Policy Assistant — Design Document

## 1. Overview

The HR Policy Assistant is a Retrieval-Augmented Generation (RAG) application that allows employees to ask questions about company policies using natural language.

The application retrieves relevant information from uploaded policy documents and generates answers using an LLM. Every answer includes citations pointing to the original document and section.

The primary design goals are:

- Answer questions using only uploaded policy documents.
- Preserve document structure during chunking.
- Provide traceable citations.
- Distinguish explicitly prohibited policies from missing information.
- Refuse questions when sufficient supporting information is unavailable.

### Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, FastAPI |
| Frontend | Streamlit |
| Embedding model | Gemini Embedding 001 |
| Vector database | ChromaDB |
| LLM | Groq |
| Response validation | JSON Schema and Python |
| Chunking | Custom Python implementation |

---

## 2. Architecture

The application consists of two main workflows: document ingestion and question answering.

### 2.1 Document Ingestion Workflow

1. An admin uploads a Markdown or TXT policy document through Streamlit.
2. FastAPI validates the file extension and content.
3. The original document is saved locally.
4. The document is divided into meaningful sections and chunks.
5. Gemini Embedding API generates an embedding for each chunk.
6. The chunks, embeddings and metadata are stored in ChromaDB.

### 2.2 Question Answering Workflow

1. An employee submits a natural-language question through Streamlit.
2. FastAPI receives the question.
3. Gemini generates an embedding for the question.
4. ChromaDB performs cosine-distance-based similarity search.
5. The top three relevant chunks are retrieved.
6. The retrieved policy context is provided to the Groq LLM.
7. The LLM determines whether the evidence supports an answer.
8. The backend validates the response and supporting evidence.
9. Citations are generated using the original document metadata.
10. The final structured JSON response is returned to Streamlit.

### Architecture Diagram

```text
              ADMIN
                |
                v
           Streamlit
                |
                v
             FastAPI
                |
                v
        Document Extraction
                |
                v
        Section-aware Chunking
                |
                v
       Gemini Embedding API
                |
                v
             ChromaDB
         (Persistent Storage)


             EMPLOYEE
                |
                v
           Streamlit
                |
                v
             FastAPI
                |
                v
       Gemini Query Embedding
                |
                v
        ChromaDB Retrieval
                |
                v
          Top 3 Chunks
                |
                v
             Groq LLM
                |
                v
      Structured JSON Response
                |
                v
        Evidence Verification
                |
                v
        Citation Generation
                |
                v
         Final Answer
```

---

## 3. Chunking and Retrieval Strategy

### 3.1 Chunking Strategy

I selected section-aware chunking instead of blindly splitting documents into fixed-size chunks.

HR policies commonly contain headings, subsections, paragraphs and tables. Preserving this structure helps maintain relationships between policy information and its original section.

The implementation follows these rules:

- Markdown headings are detected using regular expressions.
- Heading hierarchy is maintained as section metadata.
- Paragraphs are separated into logical blocks.
- Consecutive Markdown table rows are preserved together.
- Related blocks are combined until the chunk reaches approximately 1200 characters.
- Oversized text blocks are split into smaller parts.

Chunk overlap is not used because the supplied documents have relatively clear section boundaries.

Tables are preserved to avoid separating column headings from their corresponding values.

### 3.2 Metadata

Each chunk stores the following metadata:

```json
{
  "document_name": "benefits-policy.md",
  "stored_name": "8b8c5020_benefits-policy.md",
  "section": "Benefits Policy > 2. Health coverage tiers",
  "chunk_index": 3
}
```

Every chunk also has a unique ID.

This metadata provides traceability and allows the application to generate citations from the original document information.

### 3.3 Embedding Generation

I use Google's Gemini Embedding 001 model to generate embeddings.

Document chunks use the RETRIEVAL_DOCUMENT task type, while employee questions use the QUESTION_ANSWERING task type.

The default embedding dimension is 3072.

Embeddings are generated through the hosted API, avoiding the need to maintain a local machine-learning model.

### 3.4 Vector Storage and Retrieval

ChromaDB is used as the vector database.

A persistent local collection stores:

- Document embeddings.
- Original chunk text.
- Document metadata.

Cosine distance is used for similarity search.

The application initially retrieves the top three chunks for each question.

ChromaDB returns the nearest available matches even when a question is unrelated to the documents. Therefore, retrieval results alone are not treated as proof that an answer exists.

---

## 4. Grounding and Hallucination Prevention

Preventing unsupported policy answers is one of the most important requirements of this application.

I use multiple safeguards.

### 4.1 Context-Based Generation

The Groq LLM receives the employee question and retrieved policy chunks.

The system prompt instructs the model to:

- Use only the supplied policy content.
- Avoid external knowledge.
- Never guess missing information.
- Return an answer only when supporting evidence exists.

### 4.2 Evidence Classification

The LLM classifies the retrieved evidence into four categories:

| Classification | Meaning | Behaviour |
|---|---|---|
| supported | Policy explicitly provides an answer | Answer |
| explicit_negative | Policy explicitly prohibits something | Answer negatively |
| not_specified | Policy explicitly says information is undefined or handled elsewhere | Refuse |
| not_found | Retrieved content does not support an answer | Refuse |

This distinction prevents the system from interpreting missing information as an explicit prohibition.

### 4.3 Evidence Verification

For an answer, the LLM must provide:

- The retrieved chunk ID.
- An exact supporting quotation.

The backend verifies that:

1. The chunk ID belongs to the retrieved results.
2. The quotation exists in the original chunk after harmless text normalization.
3. The source chunk contains valid document and section metadata.
4. Every submitted evidence item passes validation.

Citations are generated by the backend using ChromaDB metadata rather than trusting document names or section names generated by the LLM.

### 4.4 Table-Based Questions

Markdown tables are treated as structured policy information.

The LLM is instructed to identify the relevant row and column and read their intersection.

For table-based answers, supporting evidence should include the relevant header and matching data row.

When a question involves additional conditions, separate supporting evidence is provided for those conditions.

### 4.5 Conditional Rules and Calculations

Some questions require combining multiple policy conditions.

For example, determining a meal allowance may require identifying an employee's band, finding the applicable allowance and checking the duration of travel.

The LLM may perform simple calculations when every required value and rule is explicitly available in the retrieved context.

If an essential value or condition is missing, the system refuses to answer.

### 4.6 Refusal Behaviour

When sufficient supporting information is unavailable, the application returns:

```json
{
  "status": "insufficient_information",
  "answer": "I don't have enough information in the uploaded policies. Please contact HR.",
  "citations": []
}
```

This ensures that unsupported questions do not receive invented policy answers.

**Limitation:** Evidence verification establishes that supporting quotations can be traced to retrieved documents. It does not mathematically guarantee that every generated answer is logically entailed by its evidence. This is why answer evaluation remains important.

---

## 5. API Design and Response Schema

FastAPI exposes the following endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| /health | GET | Check application health |
| /documents/upload | POST | Upload a policy document |
| /documents/{stored_name}/index | POST | Process and index a document |
| /documents/{stored_name}/chunks | GET | Inspect generated chunks |
| /retrieve | POST | Retrieve relevant policy chunks |
| /ask | POST | Generate a grounded answer |
| /vector-store/stats | GET | Check vector database statistics |

### 5.1 Ask API

Request:

```json
{
  "question": "How many casual leave days can I carry forward?"
}
```

Successful response:

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

### 5.2 Why Structured JSON?

Instead of returning only free-form text, the backend returns a predictable response containing:

- Status.
- Answer.
- Citations.

This makes it easier for the frontend to distinguish successful answers from refusals and display citations consistently.

The LLM response uses a structured schema, followed by additional backend validation.

---

## 6. Design Trade-offs

### 6.1 Custom Python Pipeline Instead of LangChain

I considered using LangChain to orchestrate the RAG workflow.

However, this application has a relatively straightforward pipeline.

I chose individual Python libraries and implemented the integration logic myself.

This provides direct control over chunking, retrieval, evidence validation and citations while keeping the implementation easier to understand and debug.

### 6.2 Hosted Embeddings Instead of Local Embeddings

I initially considered Sentence Transformers for local embedding generation.

However, local model installation introduced large machine-learning dependencies.

I selected Gemini Embedding API instead because it simplifies deployment and avoids maintaining a local embedding model.

The trade-off is dependence on an external API, internet connectivity and provider quotas.

### 6.3 ChromaDB Instead of a Managed Vector Database

I selected ChromaDB because it provides persistent vector storage and similarity search without requiring a separate managed database service.

For the relatively small document collection in this assignment, a local persistent database is sufficient.

### 6.4 Section-Aware Chunking Instead of Fixed-Size Chunking

Fixed-size chunking can separate headings, related paragraphs and table information.

Section-aware chunking preserves the logical organization of policy documents and makes citations easier to construct.

The trade-off is that unusually long sections or tables may exceed the configured chunk-size limit.

---

## 7. Evaluation

I created an automated evaluation dataset containing seven test cases.

It covers:

1. Casual leave carry-forward.
2. Standard dental coverage.
3. Plus dental coverage.
4. Lost laptop reporting.
5. Home gym reimbursement.
6. Internet reimbursement.
7. Maternity leave.

The evaluation checks:

- Response status.
- Expected answer information.
- Citation document.
- Citation section.
- Unexpected citations in refusal responses.

In addition, I manually stress-tested the application using a separate travel and expense policy containing 20 questions.

These tests covered:

- Table interpretation.
- Numerical calculations.
- Conditional rules.
- Approval requirements.
- Expense restrictions.
- Explicit negative policies.
- Missing information.
- Evidence and citation correctness.

All 20 stress-test questions passed in the completed test run.

The evaluation set is an initial regression test suite, not a comprehensive measurement of production accuracy.

---

## 8. What I Would Improve With Two More Weeks

If given two additional weeks, I would prioritize the following improvements.

### 8.1 Authentication and Authorization

Introduce proper authentication and separate admin and employee permissions.

The current interface separates the two workflows visually but does not enforce authentication or authorization.

### 8.2 Retrieval Evaluation and Improvements

Expand the evaluation dataset with more questions and expected citations.

Evaluate retrieval quality, answer correctness and refusal behaviour.

Explore hybrid search combining keyword-based retrieval with vector similarity search.

### 8.3 Document Management

Introduce policy versioning and document replacement.

When a policy is updated, its corresponding chunks should be re-indexed without retaining outdated information.

### 8.4 Improved Document Processing

Add PDF support and improve extraction of complex tables.

### 8.5 Reliability and Security

Introduce API rate limiting, improved logging, monitoring and more comprehensive error handling.

Review external embedding and LLM provider data-handling requirements before processing confidential company policies.

---

## 9. Conclusion

The HR Policy Assistant demonstrates an end-to-end RAG application with document ingestion, section-aware chunking, vector retrieval, grounded answer generation, evidence verification and citations.

The main design priority is not simply generating answers, but ensuring that answers are supported by uploaded policy information.

The implementation is intentionally kept lightweight so that its components, design choices and limitations remain understandable and maintainable.