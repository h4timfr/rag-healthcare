"""Page-aware PDF text extraction using PyMuPDF."""

from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from ..exceptions import EmptyDocumentError, InvalidPDFError, PDFExtractionError

# Above this mean token length, a page's word boundaries are gone.
#
# Measured on this project's three indexed documents: the two that extract
# correctly average 4.4-7.0 characters per token, and one whose PDF has no
# usable space information averages 53-86 - every page of it comes out as
# "Iconclude,therefore,thatthestandardofcarewasnotmetinthiscase". There is
# no ambiguity between those two populations, so the threshold sits far from
# both rather than being tuned.
#
# The consequences are silent and severe, which is why this is detected
# rather than left to be noticed in an answer: FTS5 cannot match a keyword
# inside a 900-character token, the word-based chunker sees a whole page as
# ~40 "words" and stops splitting it, and the model is asked to read a wall
# of concatenated text and then blamed for saying the information is not
# there.
MAX_MEAN_WORD_LENGTH = 15.0

# Below this many characters, a page is too short for the mean to mean
# anything (a heading, a footer, a stray line).
#
# Measured in characters rather than tokens on purpose: having very few
# tokens IS the symptom, so a token-count guard would skip exactly the worst
# pages - a page extracted as one 3000-character run is one token.
_MIN_CHARS_TO_JUDGE = 200


def mean_word_length(text: str) -> float:
    """Mean character length of whitespace-separated tokens. 0.0 for no text."""
    tokens = (text or "").split()
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def degraded_pages(
    pages: list[dict],
    max_mean_word_length: float = MAX_MEAN_WORD_LENGTH,
) -> list[int]:
    """Page numbers whose extracted text has lost its word boundaries.

    This reports; it does not repair. Reconstructing word boundaries from
    glyph geometry is a real change to what gets indexed, and it cannot be
    validated without the PDF that fails - so the honest behaviour is to
    surface the problem to whoever can re-supply the document.
    """
    flagged = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if len(text) < _MIN_CHARS_TO_JUDGE:
            continue
        if mean_word_length(text) > max_mean_word_length:
            flagged.append(page["page"])
    return flagged


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
