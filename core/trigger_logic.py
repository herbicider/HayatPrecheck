"""Pure decision logic for trigger detection and Rx-number validation.

Kept free of screen, OCR and Tk imports so it can be unit-tested anywhere. The
verification controller feeds it OCR text and acts on what it returns.
"""

import re
from typing import Iterable, List, Optional, Tuple

from rapidfuzz import fuzz

# Ordered most- to least-specific. The last one catches a bare number.
RX_PATTERNS = (r"rx\s*[#\-]\s*(\d+)", r"rx\s+(\d+)", r"(\d{4,})")

# classify_rx_reading() verdicts
RX_OK = "ok"
RX_PARTIAL = "partial"          # tail of an Rx we already know: a UI shift artifact
RX_UNCONFIRMED = "unconfirmed"  # plausible but odd; trust it only once it holds still

FULL_PHRASE_SIMILARITY = 80


def trigger_text_matches(
    text: str,
    keywords: List[str],
    similarity_threshold: float = 90,
    min_matches: int = 2,
) -> bool:
    """Does OCR text from the trigger region contain the trigger keywords?

    A keyword may be several words ("Zoom Select"). It counts as found when every
    one of its words is in the text, or the phrase appears inside longer text.
    """
    text_lower = (text or "").lower().strip()
    if not text_lower or not keywords:
        return False

    full_phrase = " ".join(keywords).lower()
    if fuzz.ratio(text_lower, full_phrase) >= FULL_PHRASE_SIMILARITY:
        return True

    text_words = [w for w in re.split(r'[\s\-_.,;:|"\']+', text_lower) if w]

    def word_found(word: str) -> bool:
        return any(fuzz.ratio(w, word) >= similarity_threshold for w in text_words)

    def keyword_found(keyword: str) -> bool:
        phrase = keyword.lower().strip()
        words = phrase.split()
        if not words:
            return False
        if all(word_found(w) for w in words):
            return True
        # partial_ratio scores a short text against any slice of the phrase, so
        # only use it when the text is at least as long as the phrase.
        return (
            len(words) > 1
            and len(text_lower) >= len(phrase)
            and fuzz.partial_ratio(phrase, text_lower) >= similarity_threshold
        )

    return sum(1 for kw in keywords if keyword_found(kw)) >= min_matches


def is_partial_rx_read(rx_number: str, reference_rx: str) -> bool:
    """Is rx_number the tail of reference_rx, as seen when the UI shifts?"""
    if not rx_number or not reference_rx:
        return False
    return (
        len(rx_number) < len(reference_rx)
        and len(rx_number) >= 3
        and reference_rx.endswith(rx_number)
    )


def classify_rx_reading(
    rx_number: str, session_rx: Optional[str], recent_rxs: Iterable[str]
) -> str:
    """Judge one OCR reading of the Rx number against what we already know."""
    recent = [rx for rx in recent_rxs if rx]

    if session_rx and is_partial_rx_read(rx_number, session_rx):
        return RX_PARTIAL
    if any(is_partial_rx_read(rx_number, rx) for rx in recent):
        return RX_PARTIAL

    if session_rx and session_rx.isdigit() and len(rx_number) != len(session_rx):
        return RX_UNCONFIRMED
    if any(len(rx_number) < len(rx) for rx in recent):
        return RX_UNCONFIRMED

    return RX_OK


def parse_rx_text(
    text: str, session_rx: Optional[str], recent_rxs: Iterable[str]
) -> Tuple[str, str]:
    """Pull an Rx number out of OCR text.

    Returns (rx_number, verdict). An RX_OK reading wins over an RX_UNCONFIRMED
    one; with neither, rx_number is "" and the verdict says why (RX_PARTIAL) or
    is "" when the text held no number at all.
    """
    text_lower = (text or "").lower()
    recent = list(recent_rxs)
    unconfirmed = ""
    saw_partial = False

    for pattern in RX_PATTERNS:
        match = re.search(pattern, text_lower)
        if not match:
            continue
        rx_number = match.group(1)
        verdict = classify_rx_reading(rx_number, session_rx, recent)
        if verdict == RX_OK:
            return rx_number, RX_OK
        if verdict == RX_UNCONFIRMED:
            unconfirmed = unconfirmed or rx_number
        else:
            saw_partial = True

    if unconfirmed:
        return unconfirmed, RX_UNCONFIRMED
    return "", RX_PARTIAL if saw_partial else ""


class StableReading:
    """Tracks how long a value has been observed without changing."""

    def __init__(self):
        self.value = ""
        self._since = 0.0

    def reset(self):
        self.value = ""
        self._since = 0.0

    def observe(self, value: str, now: float) -> float:
        """Record a reading; returns the seconds it has held (0 for no value)."""
        if not value:
            self.reset()
            return 0.0
        if value != self.value:
            self.value = value
            self._since = now
        return now - self._since
