import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

load_dotenv()


class ChunkingAndEmbeddingService:
    """
    Handles context-preserving document chunking and vector embedding generation.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        embedding_model: str = "text-embedding-3-small"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model

        # Initialize recursive character text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize OpenAI Client
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key and api_key != "your_openai_api_key_here" else None

    def chunk_documents(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits parsed page contents into chunk payloads while preserving & enriching metadata.
        """
        all_chunks = []
        global_chunk_idx = 0

        for page_data in parsed_pages:
            content = page_data.get("content", "").strip()
            page_metadata = page_data.get("metadata", {})

            if not content:
                continue

            # Split text using the configured splitter
            raw_splits = self.text_splitter.split_text(content)

            for local_idx, chunk_text in enumerate(raw_splits):
                chunk_id = f"{page_metadata.get('source_file', 'doc')}_p{page_metadata.get('page_number', 1)}_c{local_idx}_{uuid.uuid4().hex[:6]}"
                
                chunk_payload = {
                    "id": chunk_id,
                    "text": chunk_text,
                    "metadata": {
                        **page_metadata,
                        "chunk_index": global_chunk_idx,
                        "chunk_char_length": len(chunk_text),
                    }
                }
                all_chunks.append(chunk_payload)
                global_chunk_idx += 1

        return all_chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates dense vector embeddings using OpenAI's text-embedding-3-small model.
        Falls back to a deterministic pseudo-vector mock if API key is not configured yet.
        """
        if not texts:
            return []

        # If real OpenAI API key is present
        if self.client:
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=self.embedding_model
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                raise RuntimeError(f"OpenAI Embedding API error: {str(e)}")

        # Fallback Mock Generator (1536 dimensions) for local testing without API credits
        print("⚠️ OPENAI_API_KEY not configured. Generating synthetic 1536-dim embeddings for testing...")
        mock_embeddings = []
        for text in texts:
            # Deterministic mock vector based on string hash length
            seed = sum(ord(c) for c in text[:20]) % 100
            vector = [(float((i + seed) % 10) / 10.0) for i in range(1536)]
            mock_embeddings.append(vector)
        return mock_embeddings

    def process_and_embed(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Full pipeline: Chunks the text and attaches embeddings to each chunk.
        """
        chunks = self.chunk_documents(parsed_pages)
        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        return chunks