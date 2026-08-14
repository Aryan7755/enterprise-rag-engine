from app.services.parser import DocumentParser
from app.services.chunker import ChunkingAndEmbeddingService
from app.services.vector_store import VectorStoreService

def test_vector_indexing_and_search():
    print("--- Running Shift 4: Vector Store & Semantic Recall Tests ---")

    # 1. Sample document content
    sample_policy = (
        "BinaryFolks Employee Handbook 2026.\n\n"
        "Chapter 1: Remote Work Policy.\n"
        "Employees are permitted to work remotely on Tuesdays and Thursdays. "
        "High-speed internet reimbursements are capped at INR 1500 per month.\n\n"
        "Chapter 2: Annual Leave Entitlement.\n"
        "Full-time developers are entitled to 24 paid annual leaves and 10 casual leaves per financial year.\n\n"
        "Chapter 3: AI Tooling Usage.\n"
        "All engineers are encouraged to leverage Claude Code, Cursor AI, and internal LLMs for development."
    )

    # 2. Parse and chunk
    parsed_pages = DocumentParser.parse("Employee_Handbook.md", sample_policy.encode("utf-8"))
    chunker = ChunkingAndEmbeddingService(chunk_size=200, chunk_overlap=30)
    embedded_chunks = chunker.process_and_embed(parsed_pages)
    print(f"✅ Prepared & embedded {len(embedded_chunks)} chunks.")

    # 3. Initialize Vector Store and Upsert
    vector_store = VectorStoreService()
    count = vector_store.batch_upsert(embedded_chunks)
    print(f"✅ Upserted {count} vectors into Vector Store.")

    # 4. Generate query embedding for a test question
    query_text = "What is the policy for annual leaves?"
    query_vector = chunker.generate_embeddings([query_text])[0]

    # 5. Perform Semantic Search
    search_results = vector_store.semantic_search(query_vector, top_k=2)
    print(f"\n🔍 Query: '{query_text}'")
    print(f"✅ Retrieved {len(search_results)} matching chunks:\n")

    for i, res in enumerate(search_results, 1):
        print(f"Match #{i} (Score: {res['score']:.4f}):")
        print(f"  Source: {res['metadata'].get('source_file')} | Page: {res['metadata'].get('page_number')}")
        print(f"  Snippet: {res['text'][:120]}...\n")

    assert len(search_results) > 0, "No results returned from vector store"
    print("🎉 Shift 4 Vector DB Integration Verified Successfully!")

if __name__ == "__main__":
    test_vector_indexing_and_search()