"""Page-aware PDF text extraction using PyMuPDF.

Extraction runs PyMuPDF's normal "text" mode first and keeps its output
whenever that output looks like prose. When it does not, the page is
re-extracted from PyMuPDF's geometry-aware word segmentation and the repair
is kept only if it measurably improves. Everything here is deterministic and
local: no OCR, no model, and no text is ever invented - the fallback only
changes where the separators go, never what the words are.

Why the fallback exists, from the document that motivated it: one of this
project's indexed reports came out as
"Thepatientwasdischargedfrom Hospitalon3/19/2006onwarfarin,sotalol,..." -
roughly forty 60-to-100 character "words" per page. The word separators had
not been lost at all; the PDF's embedded font maps its space glyph to
character code 2 and its hyphen to code 3, and its ToUnicode table does not
say so, so extraction faithfully returned U+0002 and U+0003. Python's
str.split() does not treat those as whitespace, so every line collapsed into
one token, which then defeated FTS5 (no keyword is findable inside a
900-character token), defeated the word-based chunker (a whole page counts as
~40 "words", so it is never split), and left the model reading a wall of
concatenated text.
"""

from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from ..exceptions import EmptyDocumentError, InvalidPDFError, PDFExtractionError

# --- quality thresholds ------------------------------------------------------
#
# Each threshold sits far from BOTH populations rather than being tuned to
# the one document that motivated this. Measured across this project's
# indexed documents, healthy pages run 4.4-7.0 characters per token, 13-18%
# whitespace, and contain no control characters whatsoever; the malformed
# ones run 53-86 characters per token, ~1% whitespace, and are ~1% control
# characters. Nothing lands between the two.

# Prose does not average more than a few characters per token in any
# language this pipeline handles.
MAX_MEAN_WORD_LENGTH = 15.0

# Whitespace as a fraction of all characters. English prose is ~15-18%;
# anything near zero means the separators are missing or are not whitespace.
MIN_WHITESPACE_RATIO = 0.05

# C0 control characters (excluding tab/newline/carriage return) have no
# legitimate place in extracted document text. Their presence at ANY
# meaningful rate means the font's character mapping is not being honoured.
# This is the signal that generalises best: it is a statement about what
# valid text is, not about any particular PDF.
MAX_CONTROL_CHAR_RATIO = 0.005

# Below this many characters a page is too short for a ratio to be
# meaningful (a heading, a footer, a stray line).
#
# Measured in characters rather than tokens on purpose: having very few
# tokens IS the symptom, so a token-count guard would skip exactly the worst
# pages - a page extracted as one 3000-character run is a single token.
_MIN_CHARS_TO_JUDGE = 200

_CONTROL_CHARS = frozenset(chr(c) for c in range(32)) - frozenset("\t\n\r")


def mean_word_length(text: str) -> float:
    """Mean character length of whitespace-separated tokens. 0.0 for no text."""
    tokens = (text or "").split()
    if not tokens:
        return 0.0
    return sum(len(token) for token in tokens) / len(tokens)


def whitespace_ratio(text: str) -> float:
    """Whitespace characters as a fraction of all characters."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isspace()) / len(text)


def control_char_ratio(text: str) -> float:
    """C0 control characters (excluding tab/newline/CR) as a fraction of all
    characters. Zero for every well-formed extraction."""
    if not text:
        return 0.0
    return sum(1 for ch in text if ch in _CONTROL_CHARS) / len(text)


def text_statistics(text: str) -> dict:
    """The measurements every quality decision here is made from, so a
    decision can always be explained by the numbers that produced it."""
    stripped = (text or "").strip()
    return {
        "chars": len(stripped),
        "tokens": len(stripped.split()),
        "mean_word_length": mean_word_length(stripped),
        "whitespace_ratio": whitespace_ratio(stripped),
        "control_char_ratio": control_char_ratio(stripped),
    }


def quality_problems(
    text: str,
    max_mean_word_length: float = MAX_MEAN_WORD_LENGTH,
) -> list[str]:
    """Names of the malformed-extraction patterns present in `text`.

    Empty list means the extraction looks like prose. Text too short to
    judge is treated as healthy rather than guessed at - flagging headings
    and footers would train the reader to ignore the warning.
    """
    stats = text_statistics(text)
    if stats["chars"] == 0:
        return ["empty"]
    if stats["chars"] < _MIN_CHARS_TO_JUDGE:
        return []

    problems = []
    if stats["control_char_ratio"] > MAX_CONTROL_CHAR_RATIO:
        problems.append("control_characters")
    if stats["mean_word_length"] > max_mean_word_length:
        problems.append("long_words")
    if stats["whitespace_ratio"] < MIN_WHITESPACE_RATIO:
        problems.append("no_whitespace")
    return problems


def degraded_pages(
    pages: list[dict],
    max_mean_word_length: float = MAX_MEAN_WORD_LENGTH,
) -> list[int]:
    """Page numbers whose text is STILL malformed after extraction has done
    what it can. A page the fallback repaired is not reported: the warning
    exists to flag documents that will answer badly, not to narrate repairs
    that worked."""
    return [
        page["page"]
        for page in pages
        if quality_problems(page.get("text") or "", max_mean_word_length)
    ]


def _reconstruct_from_words(page) -> str:
    """Re-extract a page from PyMuPDF's word segmentation.

    "words" mode segments by glyph geometry rather than by trusting the
    font's character mapping, so it recovers the word boundaries that the
    broken mapping hid. Words are regrouped into their original blocks and
    lines, so page structure and reading order survive.

    This cannot invent text: every word returned here is one PyMuPDF already
    found on the page. When a page genuinely has no separable words - glyphs
    printed with no gaps at all - this returns them just as run together as
    before, and the caller keeps the original rather than pretending.
    """
    words = page.get_text("words")
    if not words:
        return ""

    lines: list[str] = []
    current: list[str] = []
    current_key = None

    # (block_no, line_no, word_no) is PyMuPDF's own reading order.
    for word in sorted(words, key=lambda w: (w[5], w[6], w[7])):
        key = (word[5], word[6])
        if key != current_key:
            if current:
                lines.append(" ".join(current))
            current = []
            current_key = key
        current.append(word[4])

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines)


def _extract_page_text(page) -> tuple[str, str, list[str]]:
    """Text for one page, plus which extraction produced it and any quality
    problems that remain. Returns (text, mode, problems)."""
    text = page.get_text("text")
    problems = quality_problems(text)
    if not problems or problems == ["empty"]:
        return text, "text", problems

    repaired = _reconstruct_from_words(page)
    repaired_problems = quality_problems(repaired)

    # Only accept the repair if it actually fixed something. A page whose
    # glyphs have no gaps is not repairable this way, and silently swapping
    # in an equally bad reconstruction would just hide that.
    if repaired and len(repaired_problems) < len(problems):
        return repaired, "words", repaired_problems

    return text, "text", problems


def extract_pages(pdf_path: str, report: dict | None = None) -> list[dict]:
    """Extract text from a PDF, one entry per non-empty page.

    Returns a list of {"page": <1-indexed page number>, "text": <page text>,
    "extraction": "text"|"words"}. The first two keys are what the rest of
    the pipeline consumes and are unchanged; "extraction" records which path
    produced the text, for the ingestion report.

    If `report` is given it is populated in place (same opt-in out-parameter
    pattern as retrieve_hybrid's `timing`) with "image_only_pages": pages
    that yielded no text but do contain an image. Those are scanned pages,
    and this pipeline has no OCR, so they are silently absent from the index
    unless someone says so - a user who uploads a 10-page scan and gets
    answers from the 2 typed pages should be told why.

    Raises InvalidPDFError, EmptyDocumentError, or PDFExtractionError on
    failure.
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
        image_only = []
        try:
            for page_number in range(doc.page_count):
                page = doc[page_number]
                text, mode, _ = _extract_page_text(page)
                if text and text.strip():
                    pages.append(
                        {"page": page_number + 1, "text": text, "extraction": mode}
                    )
                elif page.get_images(full=True):
                    image_only.append(page_number + 1)
        except InvalidPDFError:
            raise
        except Exception as e:
            raise PDFExtractionError(f"Failed extracting text from '{pdf_path}': {e}") from e
    finally:
        doc.close()

    if report is not None:
        report["image_only_pages"] = image_only

    if not pages:
        raise EmptyDocumentError(
            f"No extractable text in '{pdf_path}' (likely empty or a scanned/image-only PDF; "
            f"OCR is not part of this pipeline)"
        )

    return pages


def repaired_pages(pages: list[dict]) -> list[int]:
    """Page numbers whose text came from the word-geometry fallback."""
    return [p["page"] for p in pages if p.get("extraction") == "words"]
