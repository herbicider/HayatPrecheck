"""Regression baseline for the legacy OCR + fuzzy matching path.

These scores were captured from the pre-refactor code and are asserted here so
the offline CPU-only method cannot drift while the UI is rebuilt around it. If a
change to the matching logic is intentional, update the expected values in the
same commit and say why.
"""

import json
import os

import pytest

from core.comparison_engine import ComparisonEngine

CONFIG_PATH = os.path.join("config", "config.json")


@pytest.fixture(scope="module")
def engine():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return ComparisonEngine(json.load(f))


# (field, entered, source, expected_score, expected_match)
BASELINE = [
    ("patient_name", "SMITH, JOHN A", "John A Smith", 100.0, True),
    ("prescriber_name", "JONES, ROBERT", "Dr. Robert Jones MD", 100.0, True),
    ("drug_name", "LISINOPRIL 10MG TAB", "Lisinopril 10 mg tablet", 100, True),
    (
        "direction_sig",
        "TAKE 1 TABLET BY MOUTH DAILY",
        "Take one tablet by mouth once daily",
        85.71428571428572,
        True,
    ),
]


@pytest.mark.parametrize("field,entered,source,expected_score,expected_match", BASELINE)
def test_baseline_scores(engine, field, entered, source, expected_score, expected_match):
    result = engine.verify_fields({field: (entered, source)})[field]
    assert result["score"] == pytest.approx(expected_score)
    assert result["match"] is expected_match


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN DEFECT, left unfixed pending sign-off because it changes clinical "
        "matching behavior. verify_fields cleans the drug text (comparison_engine.py:363) "
        "and then passes the cleaned text into _enhanced_drug_name_match, which cleans it "
        "a second time (:94). The second pass expands 'mg' to 'milligram', so the dosage "
        "regex at :101 matches nothing and the dosage-mismatch guard at :105 never fires; "
        "both strings also converge on the filler 'milligram tablet', lifting the fuzzy "
        "score from 67.9 (correct reject) to 88.0 (false match at an 85 threshold). "
        "Fix: pass entered_text/source_text raw at :384, as patient_name and "
        "prescriber_name already do at :386 and :388."
    ),
)
def test_mismatch_is_flagged(engine):
    """A genuinely different drug must not pass. See xfail reason above."""
    result = engine.verify_fields(
        {"drug_name": ("METFORMIN 500MG TAB", "Lisinopril 10 mg tablet")}
    )["drug_name"]
    assert result["match"] is False


def test_double_cleaning_is_what_breaks_the_dosage_guard(engine):
    """Pin the root cause so the fix can be verified directly.

    Called with raw text the matcher rejects correctly; called with pre-cleaned
    text (what verify_fields actually does) it accepts a different drug.
    """
    entered, source = "METFORMIN 500MG TAB", "Lisinopril 10 mg tablet"

    raw_score, raw_match = engine._enhanced_drug_name_match(entered, source, 85)
    assert raw_match is False
    assert raw_score < 85

    pre_cleaned_score, pre_cleaned_match = engine._enhanced_drug_name_match(
        engine._clean_drug_name(entered), engine._clean_drug_name(source), 85
    )
    assert pre_cleaned_match is True  # the defect
    assert pre_cleaned_score > raw_score


def test_clean_drug_name_is_not_idempotent(engine):
    """'mg' -> 'milligram' on the second pass is the mechanism behind the defect."""
    once = engine._clean_drug_name("METFORMIN 500MG TAB")
    twice = engine._clean_drug_name(once)
    assert once != twice
    assert "500 mg" in once
    assert "500 milligram" in twice


def test_empty_ocr_does_not_match(engine):
    """Blank OCR output must never be treated as a match."""
    result = engine.verify_fields({"patient_name": ("", "")})["patient_name"]
    assert result["match"] is False


def test_thresholds_come_from_config(engine):
    result = engine.verify_fields({"patient_name": ("A", "B")})["patient_name"]
    assert result["threshold"] == engine.config["thresholds"]["patient"]
