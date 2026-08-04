"""Tests for the defaults schema and config round-tripping."""

import json
import os

import pytest

from core.settings_manager import DEFAULT_CONFIG, SettingsManager, config_value

REAL_CONFIG = os.path.join("config", "config.json")


def test_lookup_prefers_the_config_over_the_default():
    config = {"timing": {"fast_polling_seconds": 9.5}}
    assert config_value(config, "timing", "fast_polling_seconds") == 9.5


def test_lookup_falls_back_to_the_default():
    assert config_value({}, "timing", "fast_polling_seconds") == (
        DEFAULT_CONFIG["timing"]["fast_polling_seconds"]
    )


def test_lookup_falls_back_for_a_partially_present_section():
    """A section present but missing one key must still resolve that key."""
    config = {"automation": {"key_on_all_match": "f9"}}
    assert config_value(config, "automation", "key_on_all_match") == "f9"
    assert config_value(config, "automation", "mode") == "preset_key"


def test_unknown_setting_raises_rather_than_returning_none():
    with pytest.raises(KeyError):
        config_value({}, "timing", "no_such_setting")


def test_explicit_default_wins_over_raising():
    assert config_value({}, "nope", "nope", default="fallback") == "fallback"


def test_falsey_configured_value_is_not_treated_as_missing():
    """0 and False are real values, not absence."""
    assert config_value({"timing": {"fast_polling_seconds": 0}}, "timing", "fast_polling_seconds") == 0
    config = {"automation": {"send_key_on_all_match": False}}
    assert config_value(config, "automation", "send_key_on_all_match") is False


def test_defaults_carry_no_regions():
    """Inventing coordinates would make an unconfigured system look ready."""
    assert "regions" not in DEFAULT_CONFIG


def test_get_default_config_returns_a_copy():
    manager = SettingsManager()
    first = manager.get_default_config()
    first["timing"]["fast_polling_seconds"] = 999
    assert manager.get_default_config()["timing"]["fast_polling_seconds"] != 999


# Sections where a default disagreeing with the shipped config is a bug, because
# the old UI would silently write its own default back over the real value and
# thereby change matching or timing behavior.
BEHAVIOUR_SECTIONS = ("timing", "thresholds", "tesseract", "easyocr", "advanced_settings")


def test_defaults_agree_with_the_shipped_config():
    """Guards against the drift that used to rewrite config.json on page load.

    Scoped to the behavioral sections. Keys outside them -- verification_method,
    automation.send_key_on_all_match -- are deliberate user choices that are
    expected to differ from the out-of-the-box default.
    """
    with open(REAL_CONFIG, encoding="utf-8") as f:
        shipped = json.load(f)

    mismatches = []

    def walk(defaults, actual, path=()):
        for key, default in defaults.items():
            if key not in actual:
                continue
            if isinstance(default, dict) and isinstance(actual[key], dict):
                walk(default, actual[key], path + (key,))
            elif not isinstance(default, (dict, list)) and default != actual[key]:
                mismatches.append(
                    f"{'.'.join(path + (key,))}: default={default!r} shipped={actual[key]!r}"
                )

    for section in BEHAVIOUR_SECTIONS:
        if section in DEFAULT_CONFIG and section in shipped:
            walk(DEFAULT_CONFIG[section], shipped[section], (section,))

    assert not mismatches, "defaults disagree with config/config.json:\n  " + "\n  ".join(
        mismatches
    )


def test_user_choice_keys_have_conservative_defaults():
    """A fresh install must not auto-press keys or assume an AI endpoint exists."""
    assert DEFAULT_CONFIG["verification_method"] == "local_ocr_fuzzy"
    assert DEFAULT_CONFIG["automation"]["send_key_on_all_match"] is False


def test_save_load_round_trip_is_lossless(tmp_path):
    path = tmp_path / "config.json"
    manager = SettingsManager(str(path))
    manager.config = manager.get_default_config()
    assert manager.save_config()

    reloaded = SettingsManager(str(path))
    assert reloaded.load_config()
    assert reloaded.config == manager.config
