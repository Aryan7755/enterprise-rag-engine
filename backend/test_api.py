import io
import json
import requests

BASE_URL = "http://localhost:8000"

def test_api_workflow():
    print("--- Running Shift 7: FastAPI Endpoints & SSE Streaming Tests ---")

    # 1. Test Health Check
    res = requests.get(f"{BASE_URL}/health")
    assert res.status_code == 200, "Health check failed"
    print("✅ GET /health endpoint is operational.")

    # 2. Test Multipart File Upload
    sample_doc_content = (
        "# Enterprise Security Policy\n\n"
        "All engineers must rotate API credentials every 90 days.\n"
        "Multi-factor authentication (MFA) is strictly required for SSH server access."
    )
    files = {
        "file": ("Security_Policy.md", io.BytesIO(sample_doc_content.encode("utf-8")), "text/markdown")
    }

    upload_res = requests.post(f"{BASE_URL}/api/upload", files=files)
    assert upload_res.status_code == 200, f"Upload failed: {upload_res.text}"
    print(f"✅ POST /api/upload succeeded: {upload_res.json()}")

    # 3. Test Query with SSE Streaming
    query_payload = {
        "query": "How often should API credentials be rotated?",
        "chat_history": []
    }

    print("\n🔍 Querying: 'How often should API credentials be rotated?'\nStreaming Response:")
    with requests.post(f"{BASE_URL}/api/query", json=query_payload, stream=True) as stream_res:
        assert stream_res.status_code == 200, "Query stream failed"
        for line in stream_res.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data: "):
                    data_obj = json.loads(decoded_line.replace("data: ", ""))
                    if data_obj.get("type") == "token":
                        print(data_obj.get("content"), end="", flush=True)
                    elif data_obj.get("type") == "metadata":
                        print(f"\n[Metadata Received - Citations: {data_obj.get('citations')}]")

    print("\n\n🎉 Shift 7 API & SSE Streaming Verified Successfully!")

if __name__ == "__main__":
    test_api_workflow()