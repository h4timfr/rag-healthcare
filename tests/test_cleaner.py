from src.ingestion.cleaner import clean_text


def test_collapses_repeated_whitespace():
    assert clean_text("a   b\t\tc") == "a b c"


def test_collapses_excess_blank_lines():
    assert clean_text("a\n\n\n\nb") == "a\n\nb"


def test_strips_null_bytes():
    assert "\x00" not in clean_text("a\x00b")


def test_strips_surrounding_whitespace():
    assert clean_text("  \n hello \n  ") == "hello"
