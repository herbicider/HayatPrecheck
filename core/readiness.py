"""Is the system configured well enough to start monitoring?

Pure logic, no UI and no I/O beyond reading the VLM config file, so it can be
unit tested and reused by any front end. Replaces the scattered per-page
warnings the Streamlit UI used to show.
"""

import json
import os
from typing import Any, Dict, List, NamedTuple

# Regions every method needs: the trigger tells us a prescription is on screen
# and rx_number de-duplicates it. Both are OCR-based even in VLM mode.
SHARED_REGIONS = ("trigger", "rx_number")

# Fields the local OCR + fuzzy method always compares.
REQUIRED_FIELDS = ("patient_name", "prescriber_name", "drug_name", "direction_sig")

FIELD_LABELS = {
    "patient_name": "Patient Name",
    "prescriber_name": "Prescriber",
    "drug_name": "Drug Name",
    "direction_sig": "Directions/Sig",
    "patient_dob": "Patient DOB",
    "patient_address": "Patient Address",
    "patient_phone": "Patient Phone",
    "prescriber_address": "Prescriber Address",
}

REGION_LABELS = {
    "trigger": "Trigger Detection",
    "rx_number": "Rx Number",
}


class Readiness(NamedTuple):
    ready: bool
    problems: List[str]
    # Which tab can fix the first problem: "regions" or "ai".
    fix_tab: str = "regions"

    @property
    def summary(self) -> str:
        if self.ready:
            return "Ready to start"
        count = len(self.problems)
        return f"{count} thing{'s' if count != 1 else ''} to set up before starting"


def _is_set(coords: Any) -> bool:
    """A region is unset if it is missing, malformed, or all zeros."""
    if not isinstance(coords, (list, tuple)) or len(coords) != 4:
        return False
    return any(int(c) != 0 for c in coords)


def enabled_fields(config: Dict[str, Any]) -> List[str]:
    """Required fields plus whichever optional fields the user turned on."""
    optional = config.get("optional_fields_enabled", {})
    return list(REQUIRED_FIELDS) + [k for k, on in optional.items() if on]


def check(config: Dict[str, Any], vlm_config_path: str = "config/vlm_config.json") -> Readiness:
    """Check the config for the currently selected verification method."""
    problems: List[str] = []
    fix_tab = "regions"

    regions = config.get("regions", {}) or {}

    # Needed by both methods.
    for name in SHARED_REGIONS:
        if not _is_set(regions.get(name)):
            problems.append(f"{REGION_LABELS[name]} region is not set")

    method = config.get("verification_method", "local_ocr_fuzzy")

    if method == "vlm_ai":
        vlm_problems = _check_vlm(vlm_config_path)
        if vlm_problems and not problems:
            fix_tab = "ai"
        problems.extend(vlm_problems)
    else:
        fields = regions.get("fields", {}) or {}
        for field in enabled_fields(config):
            label = FIELD_LABELS.get(field, field.replace("_", " ").title())
            spec = fields.get(field)
            if not spec:
                problems.append(f"{label} is not configured")
                continue
            for side, side_label in (("entered", "entered"), ("source", "source")):
                if not _is_set(spec.get(side)):
                    problems.append(f"{label} — {side_label} region is not set")

    return Readiness(ready=not problems, problems=problems, fix_tab=fix_tab)


def _check_vlm(vlm_config_path: str) -> List[str]:
    problems: List[str] = []

    if not os.path.exists(vlm_config_path):
        return ["VLM config file is missing — set up a profile on the AI tab"]

    try:
        with open(vlm_config_path, "r", encoding="utf-8") as f:
            vlm_config = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return [f"VLM config could not be read ({e})"]

    regions = vlm_config.get("vlm_regions", {}) or {}
    if not any(_is_set(coords) for coords in regions.values()):
        problems.append("VLM comparison region is not set")

    profile_name = vlm_config.get("current_profile")
    profile = (vlm_config.get("profiles", {}) or {}).get(profile_name)
    if not profile:
        problems.append("No VLM profile is selected on the AI tab")
        return problems

    if not profile.get("base_url"):
        problems.append(f"VLM profile '{profile_name}' has no endpoint URL")
    if not profile.get("model_name"):
        problems.append(f"VLM profile '{profile_name}' has no model name")

    # A ${VAR} reference is only a problem when .env has no value for it. A blank
    # key is fine for local servers that do not authenticate.
    api_key = str(profile.get("api_key", ""))
    if api_key.startswith("${") and api_key.endswith("}"):
        var_name = api_key[2:-1]
        # Imported lazily so this module stays dependency-free and unit testable.
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:
            pass
        if not os.getenv(var_name):
            problems.append(
                f"API key {var_name} is empty in .env — enter it on the AI tab"
            )

    return problems
