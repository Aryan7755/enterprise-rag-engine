import re
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

# Attempt FlashRank import for cross-encoder re-ranking
try:
    from flashrank import Ranker, RerankRequest
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False


class HybridRetrieverService:
    """
    Executes hybrid search (BM25 lexical + dense vector) combined via
    Reciprocal Rank Fusion (RRF), followed by Cross-Encoder Re-ranking with FlashRank.
    """

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k
        self.ranker = None

        if FLASHRANK_AVAILABLE:
            try:
                # Ultra-lite ~4MB default cross-encoder model
                self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
                print("✅ FlashRank Cross-Encoder initialized.")
            except Exception as e:
                print(f"⚠️ FlashRank init note ({e}). Using pure RRF fallback.")
                self.ranker = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace and punctuation-safe tokenizer."""
        return re.findall(r"\w+", text.lower())

    def bm25_search(self, query: str, corpus_chunks: List[Dict[str, Any]], top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Performs BM25 lexical keyword matching over the chunk corpus.
        """
        if not corpus_chunks:
            return []

        tokenized_corpus = [self._tokenize(chunk.get("text", "")) for chunk in corpus_chunks]
        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0:  # Only return chunks with at least 1 keyword match
                results.append({
                    "id": corpus_chunks[idx]["id"],
                    "text": corpus_chunks[idx]["text"],
                    "metadata": corpus_chunks[idx].get("metadata", {}),
                    "bm25_score": float(scores[idx]),
                })
        return results

    def reciprocal_rank_fusion(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Fuses dense vector search and BM25 lexical search using Reciprocal Rank Fusion (RRF).
        Formula: RRF_Score = SUM( 1 / (k + rank) )
        """
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # 1. Score vector results
        for rank, item in enumerate(vector_results):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            chunk_map[cid] = item

        # 2. Score BM25 results
        for rank, item in enumerate(bm25_results):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = item

        # 3. Sort by aggregated RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        fused_results = []
        for cid in sorted_ids[:top_k]:
            item = chunk_map[cid]
            fused_results.append({
                "id": cid,
                "text": item.get("text", ""),
                "metadata": item.get("metadata", {}),
                "rrf_score": rrf_scores[cid]
            })

        return fused_results

    def rerank(self, query: str, candidate_chunks: List[Dict[str, Any]], final_top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Applies a cross-encoder model to score query-chunk pairs directly and selects final top-k.
        """
        if not candidate_chunks:
            return []

        if self.ranker:
            try:
                passages = [
                    {"id": chunk["id"], "text": chunk["text"], "meta": chunk.get("metadata", {})}
                    for chunk in candidate_chunks
                ]
                rerank_request = RerankRequest(query=query, passages=passages)
                ranked_output = self.ranker.rerank(rerank_request)

                final_results = []
                for item in ranked_output[:final_top_k]:
                    final_results.append({
                        "id": item["id"],
                        "text": item["text"],
                        "metadata": item.get("meta", {}),
                        "rerank_score": float(item["score"])
                    })
                return final_results
            except Exception as e:
                print(f"⚠️ FlashRank runtime error ({e}). Returning pure RRF candidates.")

        # Fallback if reranker is not active
        return candidate_chunks[:final_top_k]

    def hybrid_retrieve_and_rerank(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        corpus_chunks: List[Dict[str, Any]],
        initial_candidates_k: int = 20,
        final_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Full End-to-End Pipeline:
        1. BM25 Search
        2. Reciprocal Rank Fusion with Vector results
        3. Cross-Encoder Re-Ranking -> Final Top 5
        """
        bm25_matches = self.bm25_search(query, corpus_chunks, top_k=initial_candidates_k)
        fused_candidates = self.reciprocal_rank_fusion(vector_results, bm25_matches, top_k=initial_candidates_k)
        return self.rerank(query, fused_candidates, final_top_k=final_top_k)