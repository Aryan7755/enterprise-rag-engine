from app.services.llm import LLMService

def test_llm_generation_and_guardrails():
    print("--- Running Shift 6: Context-Engineered Prompt & LLM Generation Tests ---")

    llm = LLMService()

    # 1. Test Query Rewriting with Chat History
    history = [
        {"role": "user", "content": "What is the remote work policy?"},
        {"role": "assistant", "content": "Remote work is permitted on Tuesdays and Thursdays."}
    ]
    follow_up = "What about internet reimbursements for it?"
    rewritten = llm.rewrite_query(follow_up, history)
    print(f"✅ Conversational Query Rewriting: '{follow_up}' -> '{rewritten}'")

    # 2. Test Grounded Response Generation with Citations
    mock_chunks = [
        {
            "id": "chunk_1",
            "text": "High-speed internet reimbursements are capped at INR 1500 per month for remote staff.",
            "metadata": {
                "source_file": "Employee_Handbook.md",
                "page_number": 1
            }
        }
    ]

    result = llm.generate_response(rewritten, mock_chunks)
    print(f"\n✅ Synthesized Answer:\n{result['answer']}")
    print(f"✅ Extracted Citations: {result['citations']}")
    assert len(result["citations"]) > 0, "Citations should be tracked"

    # 3. Test Anti-Hallucination Fallback with Empty Context
    empty_result = llm.generate_response("What is the company valuation?", [])
    print(f"\n✅ Anti-Hallucination Check (Empty Context):\n{empty_result['answer']}")
    assert "cannot find sufficient information" in empty_result["answer"]

    print("\n🎉 Shift 6 LLM Generation & Prompt Engineering Verified Successfully!")

if __name__ == "__main__":
    test_llm_generation_and_guardrails()