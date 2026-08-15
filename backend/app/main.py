import os
import json
import asyncio
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from app.services.parser import DocumentParser, DocumentParsingError
from app.services.chunker import ChunkingAndEmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retriever import HybridRetrieverService
from app.services.llm import LLMService

load_dotenv()

app = FastAPI(title="Enterprise Multimodal Knowledge Engine", version="1.0.0")

# Enable CORS for Vite React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services & In-Memory Corpus Tracker
chunker_service = ChunkingAndEmbeddingService()
vector_store = VectorStoreService()
retriever_service = HybridRetrieverService()
llm_service = LLMService()

# Global in-memory list tracking all raw chunks for BM25 hybrid search
corpus_chunks_db: List[dict] = []


# --- Pydantic Schemas ---
class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: Optional[List[ChatMessage]] = None
    top_k: Optional[int] = 5


# --- Health & Status Routes ---
@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Enterprise Knowledge Engine API is running",
        "total_chunks_indexed": len(corpus_chunks_db)
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "pinecone_index": os.getenv("PINECONE_INDEX_NAME", "Not Set"),
        "openai_configured": bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_API_KEY") != "your_openai_api_key_here"),
        "indexed_chunks": len(corpus_chunks_db)
    }


# --- Endpoint 1: Document Upload & Ingestion ---
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts PDF, TXT, or MD files, parses pages with metadata,
    chunks the content, generates vector embeddings, and indexes them.
    """
    try:
        file_bytes = await file.read()
        if not file_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # 1. Parse and extract metadata
        parsed_pages = DocumentParser.parse(file.filename, file_bytes)

        # 2. Chunk and embed
        embedded_chunks = chunker_service.process_and_embed(parsed_pages)

        # 3. Batch upsert into Vector Store
        upserted_count = vector_store.batch_upsert(embedded_chunks)

        # 4. Save to BM25 corpus registry
        global corpus_chunks_db
        corpus_chunks_db.extend(embedded_chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_parsed": len(parsed_pages),
            "chunks_created": len(embedded_chunks),
            "vectors_indexed": upserted_count
        }

    except DocumentParsingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


# --- Endpoint 2: Hybrid Query with SSE Streaming ---
@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    """
    Retrieves top relevant chunks via Hybrid Search + FlashRank Reranker
    and streams the synthesized, cited response using Server-Sent Events (SSE).
    """
    raw_query = request.query
    history = [msg.model_dump() for msg in request.chat_history] if request.chat_history else []

    # 1. Rewrite conversational query
    standalone_query = llm_service.rewrite_query(raw_query, history)

    # 2. Dense Vector Retrieval
    query_vector = chunker_service.generate_embeddings([standalone_query])
    vector_results = []
    if query_vector:
        vector_results = vector_store.semantic_search(query_vector[0], top_k=20)

    # 3. Hybrid BM25 Fusion & FlashRank Re-ranking
    ranked_chunks = retriever_service.hybrid_retrieve_and_rerank(
        query=standalone_query,
        vector_results=vector_results,
        corpus_chunks=corpus_chunks_db,
        initial_candidates_k=20,
        final_top_k=request.top_k or 5
    )

    async def event_generator():
        # First event: Send retrieval metadata and source citations
        citations = list({
            f"{c['metadata'].get('source_file')} (Page {c['metadata'].get('page_number', 1)})"
            for c in ranked_chunks if "metadata" in c
        })

        meta_payload = {
            "type": "metadata",
            "rewritten_query": standalone_query,
            "citations": citations,
            "chunks_count": len(ranked_chunks)
        }
        yield f"data: {json.dumps(meta_payload)}\n\n"

        # Generate response
        response_data = llm_service.generate_response(standalone_query, ranked_chunks, history)
        answer_text = response_data.get("answer", "")

        # Stream response token-by-token or word-by-word
        words = answer_text.split(" ")
        for word in words:
            token_payload = {"type": "token", "content": word + " "}
            yield f"data: {json.dumps(token_payload)}\n\n"
            await asyncio.sleep(0.02)  # Smooth streaming latency

        # Final event: Done signal
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")