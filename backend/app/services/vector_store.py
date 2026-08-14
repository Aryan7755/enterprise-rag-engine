import os
import math
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

# Attempt to import Pinecone client
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


class InMemoryVectorStore:
    """
    In-memory fallback vector store using Cosine Similarity for local testing.
    """
    def __init__(self):
        self.vectors: Dict[str, Dict[str, Any]] = {}

    def upsert(self, records: List[Dict[str, Any]]) -> int:
        for rec in records:
            self.vectors[rec["id"]] = rec
        return len(records)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = math.sqrt(sum(a * a for a in vec1))
        norm_b = math.sqrt(sum(b * b for b in vec2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, vector: List[float], top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        scored = []
        for vid, rec in self.vectors.items():
            # Check metadata filter if supplied
            if filter_metadata:
                match = all(rec["metadata"].get(k) == v for k, v in filter_metadata.items())
                if not match:
                    continue
            score = self._cosine_similarity(vector, rec["values"])
            scored.append({
                "id": vid,
                "score": score,
                "metadata": rec["metadata"]
            })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]


class VectorStoreService:
    """
    Manages vector indexing, batch upserts, and semantic similarity search with Pinecone.
    """

    def __init__(self):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "enterprise-knowledge-base")
        self.dimension = 1536
        self.metric = "cosine"
        self.is_live = False
        self.index = None
        self.fallback_store = InMemoryVectorStore()

        self._initialize_client()

    def _initialize_client(self):
        if PINECONE_AVAILABLE and self.api_key and self.api_key != "your_pinecone_api_key_here":
            try:
                self.pc = Pinecone(api_key=self.api_key)
                # Check if index exists, else create serverless index
                existing_indexes = [idx.name for idx in self.pc.list_indexes()]
                if self.index_name not in existing_indexes:
                    print(f"Creating serverless Pinecone index '{self.index_name}'...")
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.metric,
                        spec=ServerlessSpec(cloud="aws", region="us-east-1")
                    )
                self.index = self.pc.Index(self.index_name)
                self.is_live = True
                print(f"✅ Connected to live Pinecone index: '{self.index_name}'")
            except Exception as e:
                print(f"⚠️ Pinecone initialization failed ({e}). Falling back to local in-memory vector store.")
                self.is_live = False
        else:
            print("ℹ️ Pinecone API key not provided. Running on Local In-Memory Vector Store.")
            self.is_live = False

    def batch_upsert(self, embedded_chunks: List[Dict[str, Any]], batch_size: int = 50) -> int:
        """
        Batch uploads chunks and vectors into Pinecone or fallback storage.
        """
        if not embedded_chunks:
            return 0

        total_upserted = 0

        if self.is_live and self.index:
            # Format vectors for Pinecone SDK
            for i in range(0, len(embedded_chunks), batch_size):
                batch = embedded_chunks[i : i + batch_size]
                vectors_payload = []
                for chunk in batch:
                    # Flatten metadata and include raw text inside metadata
                    meta = {**chunk["metadata"], "text": chunk["text"]}
                    vectors_payload.append({
                        "id": chunk["id"],
                        "values": chunk["embedding"],
                        "metadata": meta
                    })
                self.index.upsert(vectors=vectors_payload)
                total_upserted += len(batch)
        else:
            # Fallback memory upsert
            records = []
            for chunk in embedded_chunks:
                records.append({
                    "id": chunk["id"],
                    "values": chunk["embedding"],
                    "metadata": {**chunk["metadata"], "text": chunk["text"]}
                })
            total_upserted = self.fallback_store.upsert(records)

        return total_upserted

    def semantic_search(self, query_vector: List[float], top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Queries top_k most similar document chunks using Cosine Similarity.
        """
        if not query_vector:
            return []

        if self.is_live and self.index:
            query_kwargs = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": True
            }
            if filter_metadata:
                query_kwargs["filter"] = filter_metadata

            response = self.index.query(**query_kwargs)
            results = []
            for match in response.get("matches", []):
                results.append({
                    "id": match["id"],
                    "score": match["score"],
                    "metadata": match.get("metadata", {}),
                    "text": match.get("metadata", {}).get("text", "")
                })
            return results
        else:
            results = self.fallback_store.query(query_vector, top_k=top_k, filter_metadata=filter_metadata)
            return [
                {
                    "id": r["id"],
                    "score": r["score"],
                    "metadata": r["metadata"],
                    "text": r["metadata"].get("text", "")
                }
                for r in results
            ]