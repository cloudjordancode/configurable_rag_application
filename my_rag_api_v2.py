"""
L5 — Connecting RAG to FastAPI  (STARTER)
==========================================
Run with:
    uvicorn my_rag_api:app --reload --port 8000

Then open http://localhost:8000/docs to test all endpoints in Swagger UI.

Prerequisites:
    pip install fastapi uvicorn chromadb requests pydantic
    A docs/ folder with text files.
    Ollama running (or handle the 503 case gracefully).

Your goal: wrap the RAG pipeline in a FastAPI application.

Endpoints to implement:
    POST /ask     — accept AskRequest, return AskResponse
    POST /ingest  — load docs/ into ChromaDB, return IngestResponse
    GET  /stats   — return document count + model name
    GET  /health  — check ChromaDB and Ollama, return status dict

Key concepts:
    CORS middleware (add before any routes):
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware,
            allow_origins=["*"], allow_credentials=True,
            allow_methods=["*"], allow_headers=["*"])

    Pydantic validation:
        class AskRequest(BaseModel):
            question: str          # Pydantic raises 422 for empty string
                                   # only if you add a validator — add one!
            n_results: int = 3
            max_distance: float = 1.2

    HTTPException for 503:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Ollama is not running")

    Checking Ollama liveness:
        GET http://localhost:11434/api/tags  — returns 200 if Ollama is up
        Catch requests.exceptions.ConnectionError to detect it's down
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
import chromadb
import requests
import os
from config import settings


app = FastAPI(title="RAG API", version="1.0")

# TODO: Add CORSMiddleware — allow all origins, methods, headers

app.add_middleware(CORSMiddleware, allow_origins=["*"],
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ── Config ─────────────────────────────────────────────────────────────────

OLLAMA_URL = settings.ollama_url
MODEL = settings.model_name
DB_PATH = settings.chroma_path

# ── ChromaDB setup ─────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# ── Pydantic schemas ───────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    n_results: int = 3
    max_distance: float = 1.2

    # TODO: Add a @field_validator("question") that raises ValueError
    #   if the question is an empty string (after stripping whitespace)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be empty")
        return value.strip()


class SourceChunk(BaseModel):
    text: str
    source: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    confidence: str


class IngestResponse(BaseModel):
    chunks_ingested: int
    message: str


# ── RAG helpers ────────────────────────────────────────────────────────────
# (Paste / adapt functions from your my_rag.py / rag_pipeline.py)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer ONLY from the provided context. "
    "If the context doesn't contain the answer, say you don't have enough "
    "information. Cite source documents by name. Keep responses under 200 words."
)


def load_documents(directory: str) -> list[dict]:
    """Load all .txt and .md files from directory, chunk by paragraph."""

    docs = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join(directory, filename), "r") as f:
                content = f.read()

            paragraphs = [
                p.strip()
                for p in content.split("\n\n")
                if p.strip()
            ]

            for i, para in enumerate(paragraphs):
                docs.append({
                    "text": para,
                    "id": f"{filename}_{i}",
                    "metadata": {
                        "source": filename,
                        "chunk_index": str(i)
                    }
                })

    return docs


def retrieve(query: str, n_results: int = 3, max_distance: float = 1.2) -> list[dict]:
    """Query ChromaDB, filter by max_distance, return chunk dicts."""

    if collection.count() == 0:
        return []

    results = collection.query(
    query_texts=[query],
    n_results=min(n_results, collection.count())
)

    chunks = []

    for i in range(len(results["documents"][0])):
        distance = results["distances"][0][i]

        if distance < max_distance:
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i]["source"],
                "distance": distance
            })

    return chunks


def compute_confidence(chunks: list[dict]) -> str:
    if not chunks:
        return "none"

    best_distance = min(chunk["distance"] for chunk in chunks)

    if best_distance < 0.5:
        return "high"
    elif best_distance < 1.0:
        return "medium"
    else:
        return "low"


def call_ollama(messages: list[dict]) -> str:
    """Call Ollama (non-streaming). Raise HTTPException 503 if unreachable."""
    # TODO: POST to Ollama
    # TODO: Catch ConnectionError → raise HTTPException(status_code=503, detail="...")
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["message"]["content"]

    except requests.exceptions.RequestException:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running"
        )


def check_ollama_health() -> bool:
    """Return True if Ollama is reachable, False otherwise."""
    # TODO: GET {OLLAMA_URL}/api/tags — return True on 200, False on ConnectionError
    try:
        response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=2
        )
        return response.status_code == 200

    except requests.exceptions.RequestException:
        return False


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Retrieve context and generate a grounded answer."""
    # TODO: retrieve() with req.n_results and req.max_distance
    # TODO: If no chunks, return AskResponse with a "no documents" message,
    #       empty sources, confidence "low"
    # TODO: Build messages (SYSTEM_PROMPT + context + question)
    # TODO: call_ollama() — may raise 503
    # TODO: Return AskResponse with answer, sources, confidence

    # Step 1: Retrieve relevant document chunks
    chunks = retrieve(
        req.question,
        req.n_results,
        req.max_distance
    )

    # Step 2: Handle missing relevant information
    if not chunks:
        if collection.count() == 0:
            raise HTTPException(
                status_code=503,
                detail="No documents ingested"
            )

        return AskResponse(
            answer="I don't have relevant information to answer that.",
            sources=[],
            confidence="none"
        )

    # Step 3: Calculate retrieval confidence
    confidence = compute_confidence(chunks)

    # Step 4: Build the context
    context = "\n\n---\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in chunks
    )

    # Step 5: Construct messages for Ollama
    messages = [
        {
            "role": "system",
            "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
        },
        {
            "role": "user",
            "content": req.question
        }
    ]

    # Step 6: Generate the answer
    answer = call_ollama(messages)

    # Step 7: Return the structured response
    return AskResponse(
        answer=answer,
        sources=[SourceChunk(**chunk) for chunk in chunks],
        confidence=confidence
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Load documents from docs/ into ChromaDB."""
    # TODO: load_documents(DOCS_DIR)
    # TODO: collection.upsert(...)
    # TODO: Return IngestResponse with chunks_ingested and a message

    docs = load_documents(DOCS_DIR)

    if not docs:
        raise HTTPException(
            status_code=400,
            detail="No documents found to ingest"
        )

    collection.upsert(
        ids=[doc["id"] for doc in docs],
        documents=[doc["text"] for doc in docs],
        metadatas=[doc["metadata"] for doc in docs]
    )

    return IngestResponse(
        chunks_ingested=len(docs),
        message=f"Successfully ingested {len(docs)} chunks."
    )


@app.get("/stats")
def stats():
    """Return document count and model info."""
    # TODO: return {"document_count": collection.count(), "model": MODEL, "db_path": DB_PATH}
    return {
        "document_count": collection.count(),
        "model": MODEL,
        "db_path": DB_PATH
    }


@app.get("/health")
def health():
    """Check ChromaDB and Ollama liveness."""
    # TODO: Try collection.count() → chromadb status "ok" or "error"
    # TODO: check_ollama_health() → ollama status "connected" or "disconnected"
    # TODO: overall status "ok" only if both are healthy
    # TODO: return {"status": ..., "chromadb": ..., "ollama": ..., "document_count": ...}
        # Check ChromaDB
    try:
        document_count = collection.count()
        chromadb_status = "ok"
    except Exception:
        document_count = 0
        chromadb_status = "error"

    # Check Ollama
    ollama_ok = check_ollama_health()

    # Determine overall health
    overall_status = (
        "ok"
        if chromadb_status == "ok" and ollama_ok
        else "error"
    )

    return {
        "status": overall_status,
        "chromadb": chromadb_status,
        "ollama": "connected" if ollama_ok else "disconnected",
        "document_count": document_count
    }
