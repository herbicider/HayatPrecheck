"""Tests for trigger detection and Rx-number validation.

The Rx cases come from verification.log: readings that the old code rejected on
every poll, leaving a prescription on screen unverified for up to three minutes.
"""

import pytest

from core.trigger_logic import (
    RX_OK,
    RX_PARTIAL,
    RX_UNCONFIRMED,
    StableReading,
    classify_rx_reading,
    parse_rx_text,
    trigger_text_matches,
)


# ----------------------------------------------------------------- trigger text

@pytest.mark.parametrize(
    "text",
    ["Zoom Select", "zoom select", "Zoom Selec", "[ Zoom Select ]", "Zoom - Select x", "Tools Zoom Select Print"],
)
def test_multi_word_keyword_is_found(text):
    assert trigger_text_matches(text, ["Zoom Select"], 90, 1)


@pytest.mark.parametrize("text", ["", "Zoom", "Select", "Print Preview", "Zoo"])
def test_multi_word_keyword_needs_every_word(text):
    assert not trigger_text_matches(text, ["Zoom Select"], 90, 1)


def test_single_word_keywords_respect_min_matches():
    keywords = ["pre", "check", "rx"]
    assert trigger_text_matches("Pre-Check Rx", keywords, 90, 2)
    assert trigger_text_matches("xx check rx yy", keywords, 90, 2)
    assert not trigger_text_matches("check this out", keywords, 90, 2)


# ------------------------------------------------------------------- Rx reading

def test_tail_of_the_current_rx_is_a_ui_shift_artifact():
    assert classify_rx_reading("9116", "659116", ["659116"]) == RX_PARTIAL
    assert parse_rx_text("9116", "659116", ["659116"]) == ("", RX_PARTIAL)


def test_tail_of_a_recently_processed_rx_is_an_artifact():
    assert classify_rx_reading("6737", None, ["656737", "659116"]) == RX_PARTIAL


@pytest.mark.parametrize(
    "reading, session_rx",
    [("4475", "714485"), ("41416", "741434"), ("7118719", "712875")],
)
def test_logged_misreads_are_unconfirmed_not_dropped(reading, session_rx):
    assert parse_rx_text(reading, session_rx, [session_rx]) == (reading, RX_UNCONFIRMED)


def test_shorter_than_a_recent_rx_is_unconfirmed():
    assert classify_rx_reading("49717", None, ["704370"]) == RX_UNCONFIRMED


def test_normal_new_rx_is_ok():
    assert parse_rx_text("656737", "659116", ["659116"]) == ("656737", RX_OK)
    assert parse_rx_text("Rx - 656737", None, []) == ("656737", RX_OK)
    assert parse_rx_text("659116", "659116", ["659116"]) == ("659116", RX_OK)


def test_text_without_a_number():
    assert parse_rx_text("", "659116", []) == ("", "")
    assert parse_rx_text("abc 12", None, []) == ("", "")


# -------------------------------------------------------------- stable reading

def test_reading_must_hold_still_to_accumulate_time():
    tracker = StableReading()
    assert tracker.observe("4475", 10.0) == 0.0
    assert tracker.observe("4475", 11.5) == 1.5
    # A different reading restarts the clock.
    assert tracker.observe("0612", 12.0) == 0.0
    assert tracker.observe("0612", 14.5) == 2.5


def test_no_reading_resets_the_clock():
    tracker = StableReading()
    tracker.observe("4475", 10.0)
    assert tracker.observe("", 11.0) == 0.0
    assert tracker.observe("4475", 12.0) == 0.0
