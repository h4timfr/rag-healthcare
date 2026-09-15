import pytest

from src.ingestion.chunker import chunk_page_text, chunk_pages


def test_single_chunk_when_shorter_than_size():
    text = "one two three"
    assert chunk_page_text(text, chunk_size=10, overlap=2) == ["one two three"]


def test_splits_into_overlapping_windows():
    words = [str(i) for i in range(10)]
    text = " ".join(words)
    chunks = chunk_page_text(text, chunk_size=4, overlap=1)
    assert chunks == [
        "0 1 2 3",
        "3 4 5 6",
        "6 7 8 9",
    ]


def test_empty_text_returns_no_chunks():
    assert chunk_page_text("   ", chunk_size=5, overlap=1) == []


def test_rejects_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_page_text("a b c", chunk_size=0, overlap=0)


def test_rejects_overlap_not_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_page_text("a b c", chunk_size=3, overlap=3)


def test_chunk_pages_preserves_page_numbers():
    pages = [{"page": 1, "text": "a b c"}, {"page": 2, "text": "d e f"}]
    chunks = chunk_pages(pages, chunk_size=3, overlap=0)
    assert [c.page for c in chunks] == [1, 2]
    assert [c.text for c in chunks] == ["a b c", "d e f"]
