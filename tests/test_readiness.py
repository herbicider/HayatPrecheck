"""Tests for the pre-flight readiness check."""

import json

import pytest

from core import readiness

GOOD_REGIONS = {
    "trigger": [10, 10, 100, 40],
    "rx_number": [10, 50, 100, 80],
    "fields": {
        name: {"entered": [1, 1, 2, 2], "source": [3, 3, 4, 4]}
        for name in readiness.REQUIRED_FIELDS
    },
}


def legacy_config(**overrides):
    config = {
        "verification_method": "local_ocr_fuzzy",
        "regions": json.loads(json.dumps(GOOD_REGIONS)),
    }
    config.update(overrides)
    return config


def test_fully_configured_legacy_setup_is_ready():
    assert readiness.check(legacy_config()).ready


def test_missing_trigger_region_is_reported():
    config = legacy_config()
    del config["regions"]["trigger"]
    state = readiness.check(config)
    assert not state.ready
    assert any("Trigger" in p for p in state.problems)


def test_all_zero_region_counts_as_unset():
    """The config seeds new fields with [0,0,0,0], which is not a real region."""
    config = legacy_config()
    config["regions"]["rx_number"] = [0, 0, 0, 0]
    state = readiness.check(config)
    assert not state.ready
    assert any("Rx Number" in p for p in state.problems)


def test_enabled_optional_field_must_be_configured():
    config = legacy_config(optional_fields_enabled={"patient_dob": True})
    state = readiness.check(config)
    assert not state.ready
    assert any("Patient DOB" in p for p in state.problems)


def test_disabled_optional_field_is_ignored():
    config = legacy_config(optional_fields_enabled={"patient_dob": False})
    assert readiness.check(config).ready


def test_half_configured_field_is_reported():
    config = legacy_config()
    config["regions"]["fields"]["drug_name"]["source"] = [0, 0, 0, 0]
    state = readiness.check(config)
    assert not state.ready
    assert any("Drug Name" in p and "source" in p for p in state.problems)


# --------------------------------------------------------------------- VLM mode


def write_vlm(tmp_path, **overrides):
    config = {
        "current_profile": "p1",
        "profiles": {
            "p1": {
                "base_url": "http://localhost:1234/v1",
                "model_name": "some-model",
                "api_key": "",
            }
        },
        "vlm_regions": {"comparison": [20, 20, 900, 700]},
    }
    config.update(overrides)
    path = tmp_path / "vlm_config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def test_vlm_ready_with_blank_key(tmp_path):
    """Local servers often need no API key, so blank must be acceptable."""
    config = legacy_config(verification_method="vlm_ai")
    state = readiness.check(config, vlm_config_path=write_vlm(tmp_path))
    assert state.ready


def test_vlm_unset_region_is_reported(tmp_path):
    config = legacy_config(verification_method="vlm_ai")
    path = write_vlm(tmp_path, vlm_regions={"comparison": [0, 0, 0, 0]})
    state = readiness.check(config, vlm_config_path=path)
    assert not state.ready
    assert any("comparison region" in p for p in state.problems)


def test_vlm_unresolved_env_key_is_reported(tmp_path, monkeypatch):
    monkeypatch.delenv("VLM_TEST_KEY", raising=False)
    config = legacy_config(verification_method="vlm_ai")
    path = write_vlm(
        tmp_path,
        profiles={
            "p1": {
                "base_url": "http://x/v1",
                "model_name": "m",
                "api_key": "${VLM_TEST_KEY}",
            }
        },
    )
    state = readiness.check(config, vlm_config_path=path)
    assert not state.ready
    assert any("VLM_TEST_KEY" in p for p in state.problems)
    assert state.fix_tab == "ai"


def test_vlm_resolved_env_key_is_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_TEST_KEY", "sk-something")
    config = legacy_config(verification_method="vlm_ai")
    path = write_vlm(
        tmp_path,
        profiles={
            "p1": {
                "base_url": "http://x/v1",
                "model_name": "m",
                "api_key": "${VLM_TEST_KEY}",
            }
        },
    )
    assert readiness.check(config, vlm_config_path=path).ready


def test_vlm_missing_file_is_reported(tmp_path):
    config = legacy_config(verification_method="vlm_ai")
    state = readiness.check(config, vlm_config_path=str(tmp_path / "nope.json"))
    assert not state.ready


def test_vlm_mode_still_requires_ocr_trigger_region(tmp_path):
    """Trigger detection is OCR-based even in VLM mode."""
    config = legacy_config(verification_method="vlm_ai")
    config["regions"]["trigger"] = [0, 0, 0, 0]
    state = readiness.check(config, vlm_config_path=write_vlm(tmp_path))
    assert not state.ready
    assert any("Trigger" in p for p in state.problems)
