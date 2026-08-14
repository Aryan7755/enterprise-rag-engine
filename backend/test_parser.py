from app.services.parser import DocumentParser, DocumentParsingError

def test_document_parser():
    print("--- Running Parser Validation Tests ---")

    # Test 1: Empty file rejection
    try:
        DocumentParser.validate_file("empty.pdf", b"")
        print("❌ Test 1 Failed (Empty file was not rejected)")
    except DocumentParsingError as e:
        print(f"✅ Test 1 Passed (Empty file rejected): {e}")

    # Test 2: Unsupported extension rejection
    try:
        DocumentParser.validate_file("test.exe", b"dummy binary data")
        print("❌ Test 2 Failed (Unsupported format was not rejected)")
    except DocumentParsingError as e:
        print(f"✅ Test 2 Passed (Unsupported format rejected): {e}")

    # Test 3: Plain text / Markdown parsing with metadata
    sample_md = b"# Enterprise Policy\nAll employees must follow data safety rules."
    result = DocumentParser.parse("sample_policy.md", sample_md)
    assert len(result) == 1
    assert result[0]["metadata"]["source_file"] == "sample_policy.md"
    assert result[0]["metadata"]["file_type"] == "md"
    print("✅ Test 3 Passed (Markdown parsed with metadata structure)")

    print("\nAll parser sanity checks passed successfully!")

if __name__ == "__main__":
    test_document_parser()