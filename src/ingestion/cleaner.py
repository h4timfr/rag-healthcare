"""Text cleaning/normalization applied to raw PyMuPDF page text before chunking."""

import re
import unicodedata

# C0/C1 control characters and the zero-width/BOM characters that PDF text
# layers carry, none of which are content. Tab, newline and carriage return
# are deliberately excluded - they are the page structure.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f​-‏﻿]")

# Ligatures and typographic punctuation that PDFs use freely. Normalising
# them means a search for "fi" or a quoted phrase matches what the user
# actually typed.
_PUNCTUATION = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", " ": " ", " ": " ",
    "…": "...",
}


def clean_text(text: str) -> str:
    """Normalize whitespace, control characters and typographic punctuation
    in extracted PDF text.

    NFKC first, which folds ligatures ("ﬁ" -> "fi") and compatibility forms
    that would otherwise make a word unsearchable in FTS5 and unmatchable
    against a user's query. Control characters are then removed rather than
    left in place: a PDF whose font encoding is broken can emit them between
    every word (see pdf_extractor), and anything that survives that far is
    not content.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    for source, replacement in _PUNCTUATION.items():
        text = text.replace(source, replacement)
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
