import io
import json
import requests

BASE_URL = "http://localhost:8000"

def run_e2e_tests():
    print("--- Running Shift 9: End-to-End Integration & Edge Case Tests ---")

    # 1. Health check
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, "Backend health check failed"
    print("✅ System Health: Operational")

    # 2. Upload sample policy document
    sample_doc = (
        "# Enterprise Remote & Equipment Policy 2026\n\n"
        "## Page 1: Hardware Allowances\n"
        "Every engineer is entitled to a yearly hardware stipend of $1,200 for ergonomic accessories and monitors.\n\n"
        "## Page 2: Working Hours\n"
        "Core collaboration hours are strictly 10:00 AM to 4:00 PM EST."
    )
    
    upload_res = requests.post(
        f"{BASE_URL}/api/upload",
        files={"file": ("Policy_2026.md", io.BytesIO(sample_doc.encode("utf-8")), "text/markdown")}
    )
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    print(f"✅ Ingestion Succeeded: {upload_res.json()['chunks_created']} chunks indexed.")

    # 3. Test In-Domain Retrieval & Citations
    query_payload_1 = {"query": "What is the hardware stipend amount?"}
    print(f"\n🔍 Query 1 (In-Domain): '{query_payload_1['query']}'")
    
    answer_tokens_1 = []
    citations_1 = []
    with requests.post(f"{BASE_URL}/api/query", json=query_payload_1, stream=True) as stream_res:
        for line in stream_res.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    payload = json.loads(decoded.replace("data: ", ""))
                    if payload.get("type") == "token":
                        answer_tokens_1.append(payload.get("content"))
                    elif payload.get("type") == "metadata":
                        citations_1 = payload.get("citations", [])

    full_answer_1 = "".join(answer_tokens_1)
    print(f"Synthesized Output: {full_answer_1}")
    print(f"Verified Citations: {citations_1}")
    assert len(citations_1) > 0, "Expected document citations"

    # 4. Test Out-of-Domain / Anti-Hallucination Edge Case
    query_payload_2 = {"query": "What is the policy regarding pet allowances and dog food subsidies?"}
    print(f"\n🔍 Query 2 (Out-of-Domain Edge Case): '{query_payload_2['query']}'")

    answer_tokens_2 = []
    with requests.post(f"{BASE_URL}/api/query", json=query_payload_2, stream=True) as stream_res:
        for line in stream_res.iter_lines():
            if line:
                decoded = line.decode("utf-8")
                if decoded.startswith("data: "):
                    payload = json.loads(decoded.replace("data: ", ""))
                    if payload.get("type") == "token":
                        answer_tokens_2.append(payload.get("content"))

    full_answer_2 = "".join(answer_tokens_2)
    print(f"Synthesized Output: {full_answer_2}")
    
    # Assert anti-hallucination guardrail triggered
    assert any(phrase in full_answer_2.lower() for phrase in ["not have sufficient information", "cannot find sufficient information"]), \
        "Failed anti-hallucination check: Model fabricated an answer instead of failing gracefully."
    print("✅ Anti-hallucination edge case passed successfully!")

    print("\n🎉 Shift 9: All Integration and Edge-Case Tests Passed!")

if __name__ == "__main__":
    run_e2e_tests()