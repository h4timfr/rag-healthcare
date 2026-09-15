"""Text cleaning/normalization applied to raw PyMuPDF page text before chunking."""

import re


def clean_text(text: str) -> str:
    """Normalize whitespace and strip null bytes from extracted PDF text."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
