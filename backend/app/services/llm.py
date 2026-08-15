import os
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class LLMService:
    """
    Handles prompt engineering, conversational query rewriting,
    anti-hallucination guardrails, and deterministic response synthesis.
    """

    SYSTEM_PROMPT = """You are an Enterprise Knowledge Engine Assistant.
Your mission is to answer user queries with absolute accuracy using ONLY the provided retrieved context.

CRITICAL CONSTRAINTS & GUARDRAILS:
1. Grounded Answers: Base your answer strictly on the provided Context passages. Do NOT extrapolate, speculate, or fabricate information.
2. Mandatory Citations: Every factual assertion MUST cite its source document and page number in the format `[Source: <filename>, Page: <page_num>]`.
3. Anti-Hallucination Fallback: If the provided context does not contain sufficient facts to answer the question with certainty, explicitly reply:
   "I cannot find sufficient information in the provided documentation to answer this question accurately."
4. Tone: Professional, direct, concise, and structured with clean markdown.
"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.1):
        self.model = model
        self.temperature = temperature
        
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key and api_key != "your_openai_api_key_here" else None

    def rewrite_query(self, current_query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """
        Rewrites a follow-up query into a standalone search query given conversational context.
        """
        if not chat_history or len(chat_history) == 0:
            return current_query

        if not self.client:
            return current_query

        history_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
        
        rewrite_prompt = f"""Given the following conversation history and a new user question, rewrite the user question into a standalone, self-contained search query. Do NOT answer the question. Only output the rewritten query.

Conversation History:
{history_summary}

User Question: {current_query}
Standalone Query:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": rewrite_prompt}],
                temperature=0.0,
                max_tokens=60
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return current_query

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved passages into numbered context blocks with metadata tags.
        """
        if not retrieved_chunks:
            return "No relevant context found."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.get("metadata", {}).get("source_file", "Unknown Source")
            page = chunk.get("metadata", {}).get("page_number", 1)
            text = chunk.get("text", "").strip()

            block = f"--- Passage {i} [Source: {source}, Page: {page}] ---\n{text}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def generate_response(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generates a grounded, cited answer using retrieved context passages.
        """
        context_str = self.format_context(retrieved_chunks)

        user_content = f"""Context Passages:
{context_str}

User Question:
{query}

Provide a well-cited answer strictly grounded in the context passages above."""

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
        ]

        if chat_history:
            messages.extend(chat_history[-4:])

        messages.append({"role": "user", "content": user_content})

        # When live OpenAI API key is present
        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
                answer = response.choices[0].message.content
                citations = list({
                    f"{c['metadata'].get('source_file')} (Page {c['metadata'].get('page_number')})"
                    for c in retrieved_chunks if "metadata" in c
                })
                return {
                    "answer": answer,
                    "citations": citations,
                    "chunks_used": len(retrieved_chunks)
                }
            except Exception as e:
                return {
                    "answer": f"Error calling LLM: {str(e)}",
                    "citations": [],
                    "chunks_used": len(retrieved_chunks)
                }

        # Offline Mock Fallback: Strict content keyword verification
        print("⚠️ OPENAI_API_KEY not configured. Evaluating grounded fallback...")
        stop_words = {
            "what", "is", "the", "for", "and", "of", "in", "to", "regarding", "about", 
            "are", "a", "an", "policy", "rules", "guidelines", "document", "subsidies", "allowances"
        }
        query_words = set(re.findall(r"\w+", query.lower())) - stop_words
        
        top_chunk = None
        if retrieved_chunks and query_words:
            for chunk in retrieved_chunks:
                chunk_words = set(re.findall(r"\w+", chunk.get("text", "").lower()))
                # Must match substantive words like 'hardware', 'stipend', 'hours', etc.
                if query_words & chunk_words:
                    top_chunk = chunk
                    break

        if top_chunk:
            source = top_chunk.get("metadata", {}).get("source_file", "doc.pdf")
            page = top_chunk.get("metadata", {}).get("page_number", 1)
            mock_answer = (
                f"Based on the enterprise documentation, {top_chunk.get('text', '').strip()} "
                f"[Source: {source}, Page: {page}]"
            )
            citations = [f"{source} (Page {page})"]
        else:
            mock_answer = "I cannot find sufficient information in the provided documentation to answer this question accurately."
            citations = []

        return {
            "answer": mock_answer,
            "citations": citations,
            "chunks_used": len(retrieved_chunks)
        }