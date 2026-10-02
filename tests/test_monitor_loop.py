"""End-to-end tests of the monitoring loop against a scripted fake screen.

OCR and the AI call are stubbed, so no display, Tesseract or endpoint is needed;
the packages the controller imports still are, hence the importorskip guards.
"""

import asyncio

import pytest

pytest.importorskip("pyautogui")
pytest.importorskip("cv2")
pytest.importorskip("pytesseract")

from PIL import Image

from core import verification_controller as vc

TRIGGER_REGION = [0, 0, 10, 10]
RX_REGION = [10, 0, 20, 10]

ALL_MATCH = {"patient": 100, "prescriber": 100, "drug": 100, "direction": 100}
MISMATCH = {"patient": 100, "prescriber": 100, "drug": 40, "direction": 100}


class FakeScreen:
    """Plays back (trigger_text, rx_text) frames, one per screenshot."""

    def __init__(self, frames):
        self.frames = frames
        self.index = -1
        self.controller = None

    def screenshot(self, *args, **kwargs):
        if self.index + 1 < len(self.frames):
            self.index += 1
        else:
            self.controller.should_stop = True
        # Distinct pixels per distinct frame content, so the OCR cache behaves as
        # it does on a real screen: identical frames are not re-read.
        shade = hash(self.frames[self.index]) % 256
        return Image.new("RGB", (20, 10), (shade, shade, shade))

    def get_text_from_region(self, screenshot, region, field_name=""):
        trigger_text, rx_text = self.frames[self.index]
        return trigger_text if list(region) == TRIGGER_REGION else rx_text


class FakeVerifier:
    def __init__(self, responses):
        self.settings = {"retry_attempts": 1, "retry_delay_seconds": 0}
        self.responses = list(responses)
        self.calls = 0

    def verify_with_vlm(self):
        self.calls += 1
        return self.responses.pop(0) if self.responses else dict(ALL_MATCH)


def run(monkeypatch, frames, responses=(), cooldown=10.0):
    screen = FakeScreen(frames)
    verifier = FakeVerifier(responses)
    events = []

    monkeypatch.setattr(vc.pyautogui, "screenshot", screen.screenshot)
    monkeypatch.setattr(vc, "get_cached_ocr_provider", lambda *a, **k: screen)

    config = {
        "verification_method": "vlm_ai",
        "timing": {
            "fast_polling_seconds": 0.01,
            "same_prescription_wait_seconds": cooldown,
            "trigger_content_load_delay_seconds": 0,
        },
        "thresholds": {"patient": 50, "prescriber": 85, "drug": 85, "sig": 80},
        "regions": {"trigger": TRIGGER_REGION, "rx_number": RX_REGION, "fields": {}},
        "automation": {"send_key_on_all_match": False},
        "advanced_settings": {
            "trigger": {
                "keywords": ["Zoom Select"],
                "min_keyword_matches": 1,
                "unconfirmed_rx_stable_seconds": 0.05,
                "rx_unreadable_warn_seconds": 0.05,
            },
        },
    }

    loop = asyncio.new_event_loop()
    try:
        controller = vc.VerificationController(
            config, loop, on_event=lambda kind, payload: events.append((kind, payload))
        )
        screen.controller = controller
        monkeypatch.setattr(controller, "_get_cached_vlm_verifier", lambda: verifier)
        loop.run_until_complete(controller.async_run())
    finally:
        loop.close()

    states = [p["state"] for kind, p in events if kind == "state"]
    return controller, verifier, states


ON = "Zoom Select"


def test_states_for_a_matching_prescription(monkeypatch):
    frames = [("", "")] * 2 + [(ON, "659116")] * 3 + [("", "")] * 2
    _, verifier, states = run(monkeypatch, frames)
    assert verifier.calls == 1
    assert states == ["waiting", "reading", "checking", "match", "waiting", "stopped"]


def test_mismatch_shows_review(monkeypatch):
    frames = [(ON, "659116")] * 3
    _, _, states = run(monkeypatch, frames, responses=[MISMATCH])
    assert states[-2:] == ["review", "stopped"]


def test_same_rx_is_verified_once(monkeypatch):
    _, verifier, _ = run(monkeypatch, [(ON, "659116")] * 10)
    assert verifier.calls == 1


def test_persistently_misread_rx_is_no_longer_ignored(monkeypatch):
    """Logged case: '4475' read for 3 minutes after Rx 714485 and never verified."""
    frames = [(ON, "714485")] * 2 + [(ON, "4475")] * 30
    controller, verifier, _ = run(monkeypatch, frames)
    assert verifier.calls == 2
    assert controller.current_session_rx == "4475"


def test_tail_of_the_current_rx_is_still_ignored(monkeypatch):
    frames = [(ON, "659116")] * 2 + [(ON, "9116")] * 30
    controller, verifier, states = run(monkeypatch, frames)
    assert verifier.calls == 1
    assert controller.current_session_rx == "659116"
    assert "unreadable" not in states


def test_one_poll_misread_is_ignored(monkeypatch):
    frames = [(ON, "659116")] * 2 + [(ON, "0612")] + [(ON, "659116")] * 5
    _, verifier, _ = run(monkeypatch, frames)
    assert verifier.calls == 1


def test_unreadable_rx_is_reported(monkeypatch):
    _, verifier, states = run(monkeypatch, [(ON, "")] * 30)
    assert verifier.calls == 0
    assert states[:3] == ["waiting", "reading", "unreadable"]


def test_failed_check_is_retried_then_succeeds(monkeypatch):
    _, verifier, states = run(monkeypatch, [(ON, "659116")] * 3, responses=[{}, ALL_MATCH])
    assert verifier.calls == 2
    assert "error" not in states and "match" in states


def test_all_zero_scores_are_rechecked(monkeypatch):
    zeros = dict.fromkeys(ALL_MATCH, 0)
    _, verifier, states = run(monkeypatch, [(ON, "659116")] * 3, responses=[zeros, ALL_MATCH])
    assert verifier.calls == 2
    assert "match" in states


def test_check_that_keeps_failing_shows_error_and_is_tried_again(monkeypatch):
    frames = [(ON, "659116")] * 20
    controller, verifier, states = run(
        monkeypatch, frames, responses=[{}, {}, ALL_MATCH], cooldown=0.05
    )
    assert states.index("error") < states.index("match")
    assert verifier.calls == 3
    assert controller.current_session_rx == "659116"
