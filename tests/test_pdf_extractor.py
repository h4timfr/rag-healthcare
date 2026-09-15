import pytest

from src.exceptions import InvalidPDFError
from src.ingestion.pdf_extractor import extract_pages


def test_missing_file_raises(tmp_path):
    with pytest.raises(InvalidPDFError):
        extract_pages(str(tmp_path / "does_not_exist.pdf"))


def test_non_pdf_file_raises(tmp_path):
    bad_file = tmp_path / "not_a_pdf.pdf"
    bad_file.write_text("this is plain text, not a PDF")
    with pytest.raises(InvalidPDFError):
        extract_pages(str(bad_file))
