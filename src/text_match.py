"""Deterministic, explainable term matching shared by the answer validator
(src/generation/validation.py) and the product QA grader
(src/evaluation/product_qa.py).

Two deliberately different strictnesses live here:

- contains_term() is MORPHOLOGY-AWARE and used for terms an answer is
  REQUIRED to contain. A model that answers "a syncopal episode" has
  correctly reported a chief complaint of "syncope"; grading that as a miss
  measures the grader's suffix handling, not the model. Being lenient here
  can only ever turn a false failure into a pass.

- contains_exact() is plain case-insensitive substring matching and is used
  for terms an answer is FORBIDDEN to contain, and for anything containing a
  digit. Being lenient there could invent a failure that did not happen. A
  measured value is compared as the literal string it is - never stemmed,
  never normalised - because the digits and the decimal point ARE the fact.

No stemming library is used on purpose: the suffix table below is short,
readable, and its behavior can be reasoned about from the source. It is a
grading aid, not a linguistics claim.
"""

import re

# Longest-first: "ically" must be tried before "ical" before "al".
#
# Note "ological" is deliberately absent. Stripping it maps "cardiological"
# to "cardi", which no longer matches "cardiology" -> "cardiolog"; letting
# "ical" do the work maps both to "cardiolog", which is the whole point.
_SUFFIXES = (
    "ically",
    "ical",
    "ally",
    "ing",
    "ies",
    "ied",
    "ous",
    "ive",
    "al",
    "ic",
    "ed",
    "es",
    "y",
    "s",
    "e",
)

_MIN_STEM = 4

_WORD = re.compile(r"[a-z0-9.\-]+")


def stem(word: str) -> str:
    """Strip at most one recognised suffix, never below _MIN_STEM characters.

    syncope -> syncop, syncopal -> syncop  (the case this exists for)
    aneurysms -> aneurysm, arteries -> arter, cardiological -> cardiolog

    A word containing a digit is returned unchanged: numbers and measured
    values must match exactly, never morphologically.
    """
    w = (word or "").lower()
    if any(ch.isdigit() for ch in w):
        return w
    for suffix in _SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= _MIN_STEM:
            return w[: -len(suffix)]
    return w


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _stems(text: str) -> list[str]:
    return [stem(token) for token in _tokens(text)]


def contains_exact(haystack: str, needle: str) -> bool:
    """Case-insensitive substring match - the strict mode."""
    return bool(needle) and needle.lower() in (haystack or "").lower()


# Cues that, immediately next to a forbidden phrase, turn it from an
# assertion into a denial of it. "n't" is checked as a word suffix (not a
# separate token) so "doesn't"/"isn't"/"wasn't" all match without listing
# every contraction.
_NEGATION_WORDS = frozenset(
    {"not", "no", "never", "none", "cannot", "unavailable", "unknown", "unclear", "nowhere"}
)
_NEGATION_WORD_PATTERN = re.compile(r"[a-z']+")
_NEARBY_WORDS = 4  # how many words on each side of a match count as "immediately next to"


def contains_unnegated(haystack: str, needle: str, window_chars: int = 40) -> bool:
    """Whether `haystack` asserts `needle`, rather than denying it.

    Built for grading forbidden-phrase violations: contains_exact("The
    patient's blood type is not mentioned.", "blood type is") is True but
    the sentence denies exactly what the phrase would otherwise assert - a
    false positive that penalized a CORRECT refusal. This finds every
    occurrence of `needle` and only counts it as a real hit if at least one
    occurrence has no negation word within _NEARBY_WORDS words before or
    after it; an occurrence that IS negated is skipped, not the whole
    haystack.

    Quotation marks around the match do not suppress it - a hallucinated
    fact quoted back is still hallucinated. A needle appearing in an
    unrelated context (e.g. "5.0" as a page number) is unaffected by this
    function either way - that is a substring-matching limitation this
    function does not attempt to fix, only negation is handled here.
    """
    haystack_l = (haystack or "").lower()
    needle_l = (needle or "").lower()
    if not needle_l:
        return False

    positions = []
    start = 0
    while True:
        idx = haystack_l.find(needle_l, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    if not positions:
        return False

    for idx in positions:
        end = idx + len(needle_l)
        before = haystack_l[max(0, idx - window_chars):idx]
        after = haystack_l[end:end + window_chars]
        nearby_words = (
            _NEGATION_WORD_PATTERN.findall(before)[-_NEARBY_WORDS:]
            + _NEGATION_WORD_PATTERN.findall(after)[:_NEARBY_WORDS]
        )
        negated = any(w in _NEGATION_WORDS or w.endswith("n't") for w in nearby_words)
        if not negated:
            return True  # at least one occurrence is a real, unnegated claim

    return False  # every occurrence of `needle` was negated


def contains_term(haystack: str, needle: str) -> bool:
    """Whether `haystack` contains `needle`, allowing morphological variants.

    Exact substring wins first (cheapest, and covers every term containing
    punctuation or digits). Otherwise the needle's stemmed word sequence is
    matched against the haystack's stemmed word sequence, so a multi-word
    term like "standard of care" still has to appear as a contiguous phrase -
    the leniency is only ever in each word's ending.
    """
    if not needle:
        return False
    if contains_exact(haystack, needle):
        return True

    # Anything with a digit is a measured value: exact match only.
    if any(ch.isdigit() for ch in needle):
        return False

    needle_stems = _stems(needle)
    if not needle_stems:
        return False

    haystack_stems = _stems(haystack)
    n = len(needle_stems)
    return any(
        haystack_stems[i : i + n] == needle_stems
        for i in range(len(haystack_stems) - n + 1)
    )


# --- document references -----------------------------------------------------

# Words that carry no identifying information in a healthcare document
# filename - "the Cardio report" and "Cardio Report Sample.pdf" refer to the
# same document, and the difference is entirely in these.
_GENERIC_FILENAME_WORDS = frozenset(
    {
        "report",
        "reports",
        "sample",
        "samples",
        "document",
        "documents",
        "doc",
        "file",
        "pdf",
        "final",
        "draft",
        "copy",
        "new",
        "old",
        "v",
        "version",
        "review",
        "reviews",
        "scan",
        "export",
        "untitled",
    }
)


def document_keywords(source: str) -> list[str]:
    """The identifying words of a document filename, in order.

    "OMC Report Sample - Cardio.pdf" -> ["omc", "cardio"]
    "Pediatric Report.pdf"           -> ["pediatric"]
    "RAG_Healthcare_Review1.pdf"     -> ["rag", "healthcare"]

    Trailing digits are stripped from each word ("Review1" -> "review") so a
    numbered filename still reduces to its generic word and is dropped.
    Falls back to every word if a filename is made entirely of generic words,
    so a document is never left with no way to be referred to.
    """
    base = re.sub(r"\.[a-z0-9]+$", "", (source or "").strip(), flags=re.IGNORECASE)
    words = [w for w in re.split(r"[^A-Za-z0-9]+", base) if w]
    normalised = [re.sub(r"\d+$", "", w).lower() for w in words]
    keywords = [w for w in normalised if w and w not in _GENERIC_FILENAME_WORDS]
    return keywords or [w for w in normalised if w]


def mentions_document(text: str, source: str) -> bool:
    """Whether `text` refers to the document `source`, by filename or by an
    equivalent natural reference.

    Requiring the literal filename is the wrong check for a generated answer:
    a model that writes "the OMC Cardiology report" and "the Pediatric
    Report" has structured its answer around the right two documents, and
    marking that as a missing section measures prompt-echoing, not answer
    quality. So the full basename matches, and so does every one of the
    filename's identifying words appearing in the text - "omc" and "cardio"
    are both in "the OMC Cardiology report", with "cardio" matching inside
    "Cardiology" as a prefix.

    ALL the identifying words are required, not a majority. With a majority,
    one word would be enough for a two-word filename, and a document called
    "RAG_Healthcare_Review1.pdf" would be credited to any answer using the
    word "healthcare" - which, in a corpus of healthcare documents, is most
    of them. Crediting a document the answer never discussed is the more
    damaging error here, because this check is what decides whether an
    each-document answer actually covered each document.
    """
    if not source:
        return False
    if contains_exact(text, source):
        return True

    base = re.sub(r"\.[a-z0-9]+$", "", source.strip(), flags=re.IGNORECASE)
    if base and contains_exact(text, base):
        return True

    keywords = document_keywords(source)
    if not keywords:
        return False

    lowered = (text or "").lower()
    return all(keyword in lowered for keyword in keywords)
