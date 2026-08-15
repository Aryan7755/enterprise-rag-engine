from app.services.parser import DocumentParser
from app.services.chunker import ChunkingAndEmbeddingService
from app.services.vector_store import VectorStoreService
from app.services.retriever import HybridRetrieverService

def test_hybrid_search_and_rerank():
    print("--- Running Shift 5: Hybrid Search & Re-Ranking Tests ---")

    # 1. Create a corpus with both conceptual & keyword-specific data
    sample_corpus = (
        "Document: Cloud Infrastructure Specs & Policies.\n\n"
        "Topic 1: Server Provisioning.\n"
        "All containerized services run on AWS Elastic Beanstalk and GCP Cloud Run for auto-scaling.\n\n"
        "Topic 2: Exact SKU and Hardware Codes.\n"
        "The primary database node runs on SKU-X992-ENTERPRISE with 128GB RAM and NVMe SSDs.\n\n"
        "Topic 3: Security & Compliance.\n"
        "Zero-trust network access (ZTNA) is enforced across all internal microservices."
    )

    # 2. Ingest, chunk, and index
    parsed = DocumentParser.parse("Infra_Specs.md", sample_corpus.encode("utf-8"))
    chunker = ChunkingAndEmbeddingService(chunk_size=150, chunk_overlap=20)
    embedded_chunks = chunker.process_and_embed(parsed)

    vector_store = VectorStoreService()
    vector_store.batch_upsert(embedded_chunks)
    print(f"✅ Indexed {len(embedded_chunks)} chunks for retrieval test.")

    # 3. Test Query featuring an EXACT code ("SKU-X992-ENTERPRISE")
    query = "What are the specs for SKU-X992-ENTERPRISE?"
    query_vec = chunker.generate_embeddings([query])[0]

    # Vector search results
    vector_matches = vector_store.semantic_search(query_vec, top_k=10)

    # 4. Execute Hybrid Retrieval & Re-ranking
    retriever = HybridRetrieverService()
    final_ranked_chunks = retriever.hybrid_retrieve_and_rerank(
        query=query,
        vector_results=vector_matches,
        corpus_chunks=embedded_chunks,
        initial_candidates_k=10,
        final_top_k=2
    )

    print(f"\n🔍 Query: '{query}'")
    print(f"✅ Final Re-Ranked Top Chunks:\n")

    for i, item in enumerate(final_ranked_chunks, 1):
        score_info = f"ReRank Score: {item.get('rerank_score', item.get('rrf_score', 0)):.4f}"
        print(f"Rank #{i} [{score_info}]:")
        print(f"  Source: {item['metadata'].get('source_file')} | Page: {item['metadata'].get('page_number')}")
        print(f"  Text: {item['text']}\n")

    # Validate that the top result contains the exact target keyword
    assert "SKU-X992-ENTERPRISE" in final_ranked_chunks[0]["text"]
    print("🎉 Shift 5 Hybrid Search & Re-ranking Logic Verified Successfully!")

if __name__ == "__main__":
    test_hybrid_search_and_rerank()