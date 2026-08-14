import os
import io
from datetime import datetime, timezone
from typing import List, Dict, Any
from pypdf import PdfReader
from pypdf.errors import PdfReadError, FileNotDecryptedError


class DocumentParsingError(Exception):
    """Custom exception raised when document parsing or validation fails."""
    pass


class DocumentParser:
    """
    Handles validation, text extraction, and metadata tagging for uploaded documents (PDF, TXT, MD).
    """

    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @staticmethod
    def validate_file(filename: str, file_bytes: bytes) -> None:
        """
        Validates file extension, size, and basic byte integrity.
        """
        _, ext = os.path.splitext(filename)
        if ext.lower() not in DocumentParser.ALLOWED_EXTENSIONS:
            raise DocumentParsingError(
                f"Unsupported file format '{ext}'. Allowed formats: {', '.join(DocumentParser.ALLOWED_EXTENSIONS)}"
            )

        if not file_bytes or len(file_bytes) == 0:
            raise DocumentParsingError("File is empty (0 bytes).")

    @classmethod
    def parse_pdf(cls, filename: str, file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extracts text from each page of a PDF file, ensuring it is unencrypted and non-corrupt.
        """
        extracted_pages = []
        upload_time = datetime.now(timezone.utc).isoformat()

        try:
            pdf_stream = io.BytesIO(file_bytes)
            reader = PdfReader(pdf_stream)

            # Check if PDF is encrypted / password-protected
            if reader.is_encrypted:
                raise DocumentParsingError(
                    f"File '{filename}' is password-protected or encrypted. Please provide an unencrypted PDF."
                )

            total_pages = len(reader.pages)
            if total_pages == 0:
                raise DocumentParsingError(f"File '{filename}' contains no readable pages.")

            for page_idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                cleaned_text = page_text.strip()

                # Preserve page metadata even if page has minimal text
                extracted_pages.append({
                    "content": cleaned_text,
                    "metadata": {
                        "source_file": filename,
                        "page_number": page_idx + 1,
                        "total_pages": total_pages,
                        "file_type": "pdf",
                        "upload_timestamp": upload_time,
                    }
                })

            return extracted_pages

        except (FileNotDecryptedError, PdfReadError) as e:
            raise DocumentParsingError(f"Corrupt or unreadable PDF '{filename}': {str(e)}")
        except Exception as e:
            if isinstance(e, DocumentParsingError):
                raise
            raise DocumentParsingError(f"Failed to process PDF '{filename}': {str(e)}")

    @classmethod
    def parse_text(cls, filename: str, file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extracts content from plain text or markdown files.
        """
        upload_time = datetime.now(timezone.utc).isoformat()
        try:
            text_content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text_content = file_bytes.decode("latin-1")
            except Exception as e:
                raise DocumentParsingError(f"Unable to decode text file '{filename}': {str(e)}")

        _, ext = os.path.splitext(filename)
        return [{
            "content": text_content.strip(),
            "metadata": {
                "source_file": filename,
                "page_number": 1,
                "total_pages": 1,
                "file_type": ext.lower().replace(".", ""),
                "upload_timestamp": upload_time,
            }
        }]

    @classmethod
    def parse(cls, filename: str, file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Main entrypoint: validates and routes to the appropriate parser based on file extension.
        """
        cls.validate_file(filename, file_bytes)
        _, ext = os.path.splitext(filename)

        if ext.lower() == ".pdf":
            return cls.parse_pdf(filename, file_bytes)
        else:
            return cls.parse_text(filename, file_bytes)