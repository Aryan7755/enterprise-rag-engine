from app.services.parser import DocumentParser
from app.services.chunker import ChunkingAndEmbeddingService

def test_chunking_and_embeddings():
    print("--- Running Shift 3: Chunking & Embedding Tests ---")

    # 1. Simulate a multi-paragraph sample document
    sample_text = (
        "BinaryFolks Enterprise Architecture Guidelines.\n\n"
        "Section 1: Data Ingestion.\n"
        "The system must handle unstructured documents, including PDFs, Markdown, and TXT files. "
        "Each document is parsed, metadata is extracted, and the content is split into context-aware chunks. " * 5
        + "\n\nSection 2: Vector Search & Embeddings.\n"
        "Chunks are converted into 1536-dimensional dense vectors using text-embedding-3-small. "
        "These vectors are indexed in a high-throughput Vector Database for semantic search." * 5
    )

    # 2. Parse mock file using Shift 2 parser
    parsed_docs = DocumentParser.parse("Architecture_Guide.md", sample_text.encode("utf-8"))
    print(f"✅ Parser Output: {len(parsed_docs)} page(s) parsed.")

    # 3. Initialize chunking and embedding service
    service = ChunkingAndEmbeddingService(chunk_size=400, chunk_overlap=80)
    processed_chunks = service.process_and_embed(parsed_docs)

    print(f"✅ Total Chunks Generated: {len(processed_chunks)}")
    assert len(processed_chunks) > 1, "Text should be split into multiple chunks"

    # 4. Verify first chunk structure & metadata
    first_chunk = processed_chunks[0]
    print(f"✅ Chunk ID Format: {first_chunk['id']}")
    print(f"✅ Metadata Verified: Source = {first_chunk['metadata']['source_file']}, Page = {first_chunk['metadata']['page_number']}")

    # 5. Verify embedding dimensionality (1536 dims standard for text-embedding-3-small)
    embedding = first_chunk.get("embedding")
    assert embedding is not None, "Embedding vector missing"
    assert len(embedding) == 1536, f"Expected 1536 dimensions, got {len(embedding)}"
    print(f"✅ Vector Embedding Verified: Dimension = {len(embedding)}")

    print("\n🎉 Shift 3 Pipeline Verified Successfully!")

if __name__ == "__main__":
    test_chunking_and_embeddings()