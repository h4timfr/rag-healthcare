"""Page-aware PDF text extraction using PyMuPDF."""

from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from ..exceptions import EmptyDocumentError, InvalidPDFError, PDFExtractionError


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract text from a PDF, one entry per non-empty page.

    Returns a list of {"page": <1-indexed page number>, "text": <raw page text>}.
    Raises InvalidPDFError, EmptyDocumentError, or PDFExtractionError on failure.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise InvalidPDFError(f"PDF not found: {pdf_path}")
    if not path.is_file():
        raise InvalidPDFError(f"Not a file: {pdf_path}")

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise InvalidPDFError(f"Could not open '{pdf_path}' as a PDF: {e}") from e

    try:
        if doc.is_encrypted:
            raise InvalidPDFError(f"PDF '{pdf_path}' is encrypted/password-protected")

        if doc.page_count == 0:
            raise EmptyDocumentError(f"PDF '{pdf_path}' has no pages")

        pages = []
        try:
            for page_number in range(doc.page_count):
                page = doc[page_number]
                text = page.get_text("text")
                if text and text.strip():
                    pages.append({"page": page_number + 1, "text": text})
        except InvalidPDFError:
            raise
        except Exception as e:
            raise PDFExtractionError(f"Failed extracting text from '{pdf_path}': {e}") from e
    finally:
        doc.close()

    if not pages:
        raise EmptyDocumentError(
            f"No extractable text in '{pdf_path}' (likely empty or a scanned/image-only PDF; "
            f"OCR is not part of this pipeline)"
        )

    return pages
