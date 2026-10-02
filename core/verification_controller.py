
import asyncio
import functools
import json
import logging
import re
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import pyautogui
from PIL import Image, ImageFilter

from core import trigger_logic
from core.comparison_engine import ComparisonEngine
from core.logger_config import log_rx_summary, setup_logging
from core.ocr_provider import get_cached_ocr_provider
from core.settings_manager import substitute_env_vars


# This function is defined at the top level so it can be pickled and sent to other processes.
def perform_ocr_task(
    ocr_provider_type: str,
    advanced_settings: Dict[str, Any],
    screenshot_bytes: bytes,
    width: int,
    height: int,
    region: Tuple[int, int, int, int],
    field_identifier: str,
) -> Tuple[str, str]:
    """
    A self-contained function to be run in a separate process for CPU-bound OCR work.
    It initializes its own OCR provider instance to ensure process safety.
    """
    try:
        screenshot = Image.frombytes("RGB", (width, height), screenshot_bytes)
        
        # Each process gets its own OCR provider. The cache will be per-process.
        ocr_provider = get_cached_ocr_provider(ocr_provider_type, advanced_settings)

        # Detect warm-up tasks by identifier
        is_warmup = str(field_identifier).startswith("warmup_")

        # Simplified synchronous retry logic for the isolated process
        retry_config = advanced_settings.get("timing", {})
        retry_delay = retry_config.get("ocr_retry_delay_seconds", 0.5)
        max_retries = 1 if is_warmup else int(retry_config.get("ocr_max_retries", 3))

        for attempt in range(max_retries):
            text = ocr_provider.get_text_from_region(screenshot, region, field_identifier)
            if text and text.strip():
                if attempt > 0 and not is_warmup:
                    logging.info(
                        f"OCR (process) success for {field_identifier} on attempt {attempt + 1}"
                    )
                return field_identifier, text
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        # For warm-up jobs, we don't treat empty OCR as an error — the goal is to load providers.
        if is_warmup:
            logging.debug(f"OCR warm-up completed for {field_identifier}")
            return field_identifier, ""

        logging.error(
            f"OCR (process) failed for {field_identifier} after {max_retries} attempts."
        )
        return field_identifier, ""

    except Exception as e:
        logging.error(f"Error in OCR process task for {field_identifier}: {e}")
        return field_identifier, ""




class VerificationController:
    """Manages the main async application loop, screen monitoring, and UI."""

    AHK_MODIFIER_MAP = {
        "^": "ctrl",
        "!": "alt",
        "+": "shift",
        "#": "win",
    }
    AHK_KEY_MAP = {
        "capslock": "capslock",
        "ctrl": "ctrl",
        "del": "delete",
        "delete": "delete",
        "down": "down",
        "end": "end",
        "enter": "enter",
        "esc": "esc",
        "escape": "esc",
        "f1": "f1",
        "f2": "f2",
        "f3": "f3",
        "f4": "f4",
        "f5": "f5",
        "f6": "f6",
        "f7": "f7",
        "f8": "f8",
        "f9": "f9",
        "f10": "f10",
        "f11": "f11",
        "f12": "f12",
        "home": "home",
        "ins": "insert",
        "insert": "insert",
        "left": "left",
        "pgdn": "pagedown",
        "pgup": "pageup",
        "right": "right",
        "space": "space",
        "tab": "tab",
        "up": "up",
    }

    def __init__(
        self,
        config: Dict[str, Any],
        loop: asyncio.AbstractEventLoop,
        on_event: Optional[Callable[[str, Any], None]] = None,
    ):
        """
        Args:
            on_event: Called from the monitoring thread as on_event(kind, payload).
                Kinds are "results" (payload: per-field verification results),
                "clear_overlay" (payload: None) and "state" (payload:
                {"state": one of the STATE_* names, "text": description}).
                The callback must be thread-safe and must not block -- the UI
                marshals these onto its own thread.
        """
        self.config = config
        self.loop = loop
        self.on_event = on_event
        self.advanced_settings = config.get("advanced_settings", {})

        self.ocr_provider_type = config.get("ocr_provider", "tesseract")
        # This provider is for quick, synchronous checks in the main loop (e.g., trigger).
        self.ocr_provider = get_cached_ocr_provider(
            self.ocr_provider_type, self.advanced_settings
        )
        logging.info(f"Using OCR provider: {self.ocr_provider_type}")
        
        self.comparison_engine = ComparisonEngine(config)

        # Configure process pool size (optional)
        startup_cfg = self.config.get("advanced_settings", {}).get("startup", {})
        ocr_worker_count = startup_cfg.get("ocr_worker_count")
        if isinstance(ocr_worker_count, int) and ocr_worker_count > 0:
            self.process_pool = ProcessPoolExecutor(max_workers=ocr_worker_count)
        else:
            self.process_pool = ProcessPoolExecutor()

        self.recently_triggered = False
        self.last_rx_number = None
        self.last_screenshot_hash = None
        self.trigger_check_count = 0  # Add counter for trigger checks
        self.verification_in_progress = False
        self.overlay_visible = False

        # Cache VLM verifier to avoid repeated initialization
        self._vlm_verifier_cache = None
        self._vlm_config_hash = None
        self.last_trigger_time = 0
        self.last_seen_trigger_time = 0.0
        self.processed_rx_times: Dict[str, float] = {}
        self.processed_rx_signatures: Dict[str, str] = {}  # Track Rx signatures to prevent duplicates
        self.current_session_rx: Optional[str] = None  # Track the current session Rx to prevent reprocessing
        self.last_verified_signature = ""
        self.overlay_created_time = 0
        self.should_stop = False
        self.skip_count_for_current_rx = 0  # Track skips for current Rx

        self._last_state: Optional[Tuple[str, str]] = None
        self._waiting_text = "Waiting for Rx"
        # Last OCR result per region, reused while the region's pixels are unchanged.
        self._region_ocr_cache: Dict[str, Tuple[bytes, str]] = {}
        # Rx reading that failed validation but may still be a real, misread Rx.
        self._unconfirmed_rx = ""
        self._unconfirmed_tracker = trigger_logic.StableReading()
        self._rx_verdict = ""
        self._rx_raw_text = ""
        self._rx_unreadable_since = 0.0
        self._rx_unreadable_logged = 0.0

        # Optional OCR warm-up to avoid first-use latency
        try:
            warm_up_main = bool(startup_cfg.get("warm_up_ocr_on_start", False))
            warm_up_workers = bool(startup_cfg.get("warm_up_ocr_workers", False))
            workers_to_warm = int(startup_cfg.get("workers_to_warm", 2))

            if warm_up_main:
                self._warm_up_main_ocr()
            # The worker pool only serves the legacy per-field OCR method.
            if warm_up_workers and self._verification_method() != "vlm_ai":
                self._warm_up_worker_processes(max(1, workers_to_warm))
        except Exception as e:
            logging.debug(f"OCR warm-up skipped due to error: {e}")

    def _verification_method(self) -> str:
        """The configured method, honoring the legacy verification_mode key."""
        if "verification_mode" in self.config and "verification_method" not in self.config:
            legacy_mode = self.config.get("verification_mode", "ocr")
            return "vlm_ai" if legacy_mode == "vlm" else "local_ocr_fuzzy"
        return self.config.get("verification_method", "local_ocr_fuzzy")

    def _get_cached_vlm_verifier(self):
        """Get or create cached VLM verifier instance to avoid repeated initialization"""
        try:
            # Load current VLM configuration
            vlm_config = self._load_vlm_config()
            if not vlm_config:
                return None
            
            # Create a hash of the VLM config to detect changes
            import hashlib
            config_str = str(sorted(vlm_config.items()))
            current_config_hash = hashlib.md5(config_str.encode()).hexdigest()
            
            # If config changed or no cached verifier, create new one
            if (self._vlm_config_hash != current_config_hash or 
                self._vlm_verifier_cache is None):
                
                try:
                    from ai.vlm_verifier import VLM_Verifier
                    self._vlm_verifier_cache = VLM_Verifier(vlm_config)
                    self._vlm_config_hash = current_config_hash
                    logging.info("VLM: Created cached verifier instance")
                except ImportError:
                    logging.warning("VLM: VLM Verifier module not available")
                    return None
            
            return self._vlm_verifier_cache
            
        except Exception as e:
            logging.error(f"Error getting cached VLM verifier: {e}")
            return None

    def _get_screenshot_hash(self, screenshot: Image.Image) -> str:
        """Get a quick hash of the screenshot to detect changes."""
        try:
            hashing_config = self.advanced_settings.get("hashing", {})
            crop_box = hashing_config.get("crop_box", {"left": 50, "top": 150, "right": 800, "bottom": 500})
            
            left = crop_box["left"]
            top = crop_box["top"]
            right = min(screenshot.width, crop_box["right"])
            bottom = min(screenshot.height, crop_box["bottom"])
            
            cropped = screenshot.crop((left, top, right, bottom))
            
            resize_to = tuple(hashing_config.get("resize_to", [32, 32]))
            small_image = cropped.convert('L').resize(resize_to)
            
            blur_radius = hashing_config.get("blur_radius", 0.5)
            small_image = small_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            pixels = list(small_image.getdata())
            
            bucket_size = hashing_config.get("bucket_size", 8)
            bucketed_pixels = [p // bucket_size * bucket_size for p in pixels]
            
            return str(hash(tuple(bucketed_pixels)))
        except Exception as e:
            logging.error(f"Error creating screenshot hash: {e}")
            return ""

    def _has_screen_changed(self, screenshot: Image.Image) -> bool:
        """Check if the screen has changed since last check."""
        current_hash = self._get_screenshot_hash(screenshot)
        if self.last_screenshot_hash is None or current_hash != self.last_screenshot_hash:
            self.last_screenshot_hash = current_hash
            return True
        return False

    def _get_prescription_signature(self, ocr_results: Dict[str, Tuple[str, str]]) -> str:
        """Get a signature of the current prescription to detect changes."""
        try:
            signature_parts = []
            
            mandatory_fields = ["patient_name", "drug_name"]
            
            for field_name in mandatory_fields:
                if field_name in ocr_results:
                    entered_text, source_text = ocr_results[field_name]
                    if field_name == "patient_name":
                        clean_entered = self.comparison_engine._normalize_name(entered_text, is_entered_field=True)
                        clean_source = self.comparison_engine._normalize_name(source_text, is_entered_field=False)
                    else:
                        clean_entered = self.comparison_engine._clean_text(entered_text)
                        clean_source = self.comparison_engine._clean_text(source_text)
                    signature_parts.append(f"{clean_entered}|{clean_source}")
            return "::".join(signature_parts)
        except Exception as e:
            logging.error(f"Error creating prescription signature: {e}")
            return ""

    def _emit(self, kind: str, payload: Any = None):
        """Hand an event to the UI. Never raises into the monitoring loop."""
        if not self.on_event:
            return
        try:
            self.on_event(kind, payload)
        except Exception as e:
            logging.error(f"Error dispatching '{kind}' event: {e}")

    def _set_state(self, state: str, text: str):
        """Tell the UI what the monitor is doing. Emits only on change."""
        if (state, text) == self._last_state:
            return
        self._last_state = (state, text)
        self._emit("state", {"state": state, "text": text})

    def _show_overlay(self, results: Dict[str, Any]):
        """Ask the UI to display the score overlay."""
        self.overlay_visible = True
        self.overlay_created_time = time.time()
        self._emit("results", results)

    def _close_overlay(self):
        """Ask the UI to hide the score overlay."""
        if not self.overlay_visible:
            return
        self.overlay_visible = False
        self._emit("clear_overlay")

    def _warm_up_main_ocr(self):
        """Preload OCR provider in the main process to avoid first-use latency."""
        try:
            from PIL import Image
            import numpy as np
            # Create a tiny dummy image and run a minimal OCR call
            dummy = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))
            provider = get_cached_ocr_provider(self.ocr_provider_type, self.advanced_settings)
            _ = provider.get_text_from_region(dummy, (0, 0, 8, 8), "warmup_main")
            logging.info("OCR warm-up (main process) completed")
        except Exception as e:
            logging.debug(f"Main OCR warm-up failed: {e}")

    def _warm_up_worker_processes(self, count: int):
        """Submit no-op OCR tasks to spin up worker processes and load models."""
        try:
            from PIL import Image
            import numpy as np
            dummy_img = Image.fromarray(np.full((16, 16, 3), 255, dtype=np.uint8))
            dummy_bytes = dummy_img.tobytes()
            width, height = dummy_img.size

            futures = []
            for i in range(count):
                fut = self.process_pool.submit(
                    perform_ocr_task,
                    self.ocr_provider_type,
                    self.advanced_settings,
                    dummy_bytes,
                    width,
                    height,
                    (0, 0, 8, 8),
                    f"warmup_worker_{i}",
                )
                futures.append(fut)

            # Wait briefly for warm-up without blocking too long
            for fut in futures:
                try:
                    fut.result(timeout=10)
                except Exception:
                    pass
            logging.info(f"OCR warm-up ({len(futures)} worker(s)) completed")
        except Exception as e:
            logging.debug(f"Worker OCR warm-up failed: {e}")

    async def _handle_all_fields_matched(self):
        """Handle when all fields match, now with async sleep."""
        automation_config = self.config.get("automation", {})
        if not automation_config.get("send_key_on_all_match"):
            return

        delay_seconds = automation_config.get("key_delay_seconds", 0.5)
        automation_mode = self._get_automation_mode(automation_config)

        logging.info(f"SUCCESS: All fields matched! Running automation mode '{automation_mode}' in {delay_seconds}s...")
        self._set_state("sending", f"Rx#{self.last_rx_number} matched — sending key")
        await asyncio.sleep(delay_seconds)

        if automation_mode == "autohotkey_v2":
            await self._run_ahk_style_automation(automation_config)
            return

        key_to_send = automation_config.get("key_on_all_match", "f12")
        try:
            # pyautogui is blocking, but short. For true async, this would also go in an executor.
            pyautogui.press(str(key_to_send).lower())
            logging.info(f"SUCCESS: Sent '{key_to_send}' key press successfully")
        except Exception as e:
            logging.error(f"Error sending key press: {e}")

    def _get_automation_mode(self, automation_config: Dict[str, Any]) -> str:
        """Resolve the configured automation mode with backward-compatible fallback."""
        mode = str(automation_config.get("mode", "preset_key")).strip().lower()
        if mode in {"preset_key", "autohotkey_v2"}:
            return mode
        logging.warning(f"Unknown automation mode '{mode}', falling back to preset_key")
        return "preset_key"

    def _tokenize_ahk_send_sequence(self, sequence: str) -> List[Tuple[str, str]]:
        """Tokenize a small AutoHotkey-style send sequence."""
        tokens: List[Tuple[str, str]] = []
        index = 0
        while index < len(sequence):
            current_char = sequence[index]
            if current_char == "{":
                closing_index = sequence.find("}", index + 1)
                if closing_index == -1:
                    raise ValueError(f"Unclosed key token in send sequence: {sequence}")
                tokens.append(("key", sequence[index + 1:closing_index]))
                index = closing_index + 1
                continue
            tokens.append(("text", current_char))
            index += 1
        return tokens

    def _parse_ahk_command_arguments(self, raw_value: str) -> str:
        """Extract the argument payload for supported one-line AHK-style commands."""
        argument_text = raw_value.strip()
        if not argument_text:
            return ""
        if argument_text.startswith('"') and argument_text.endswith('"') and len(argument_text) >= 2:
            return argument_text[1:-1]
        return argument_text

    def _normalize_ahk_key(self, raw_key: str) -> str:
        """Map a small AHK-style key token to a pyautogui key."""
        normalized_key = raw_key.strip().lower()
        if normalized_key in self.AHK_KEY_MAP:
            return self.AHK_KEY_MAP[normalized_key]
        if len(normalized_key) == 1:
            return normalized_key
        raise ValueError(f"Unsupported AHK key token: {raw_key}")

    def _execute_ahk_send(self, sequence: str):
        """Interpret a subset of AHK Send syntax and replay it with pyautogui."""
        modifier_stack: List[str] = []
        index = 0

        while index < len(sequence):
            current_char = sequence[index]
            if current_char in self.AHK_MODIFIER_MAP:
                modifier_stack.append(self.AHK_MODIFIER_MAP[current_char])
                index += 1
                continue

            if current_char == "{":
                closing_index = sequence.find("}", index + 1)
                if closing_index == -1:
                    raise ValueError(f"Unclosed key token in send sequence: {sequence}")
                key_token = sequence[index + 1:closing_index]
                normalized_key = self._normalize_ahk_key(key_token)
                if modifier_stack:
                    pyautogui.hotkey(*modifier_stack, normalized_key)
                    modifier_stack = []
                else:
                    pyautogui.press(normalized_key)
                index = closing_index + 1
                continue

            if modifier_stack:
                pyautogui.hotkey(*modifier_stack, current_char.lower())
                modifier_stack = []
            else:
                pyautogui.write(current_char)
            index += 1

        if modifier_stack:
            raise ValueError("Dangling modifier in AHK send sequence")

    def _execute_ahk_style_script(self, automation_config: Dict[str, Any]) -> bool:
        """Interpret a small AHK-style script internally without external AutoHotkey."""
        script_code = str(automation_config.get("autohotkey_v2_code", "")).strip()
        if not script_code:
            logging.error("AHK-style automation is enabled but no script code is configured")
            return False

        try:
            for line_number, raw_line in enumerate(script_code.splitlines(), start=1):
                stripped_line = raw_line.strip()
                if not stripped_line or stripped_line.startswith(";"):
                    continue
                if stripped_line.startswith("#"):
                    continue

                command_match = re.match(r"^(SendText|Send|Sleep)\s+(.*)$", stripped_line, re.IGNORECASE)
                if not command_match:
                    raise ValueError(f"Unsupported AHK-style command on line {line_number}: {stripped_line}")

                command_name = command_match.group(1).lower()
                command_args = self._parse_ahk_command_arguments(command_match.group(2))

                if command_name == "sleep":
                    time.sleep(float(command_args) / 1000.0)
                elif command_name == "sendtext":
                    pyautogui.write(command_args)
                else:
                    self._execute_ahk_send(command_args)

            logging.info("SUCCESS: Parsed AHK-style automation script executed successfully")
            return True
        except Exception as e:
            logging.error(f"Error executing parsed AHK-style automation: {e}")
            return False

    async def _run_ahk_style_automation(self, automation_config: Dict[str, Any]):
        """Execute parsed AHK-style automation without blocking the event loop."""
        try:
            success = await asyncio.to_thread(
                self._execute_ahk_style_script,
                dict(automation_config),
            )
            if not success:
                logging.error("Parsed AHK-style automation did not complete successfully")
        except Exception as e:
            logging.error(f"Error scheduling parsed AHK-style automation: {e}")

    def _is_partial_rx_read(self, rx_number: str, reference_rx: str) -> bool:
        """Check if rx_number is likely a partial read of reference_rx due to UI shifts."""
        return trigger_logic.is_partial_rx_read(rx_number, reference_rx)

    def _ocr_region_cached(
        self, key: str, screenshot: Image.Image, region: tuple, read: Callable[[], str]
    ) -> str:
        """OCR a region, reusing the last result while its pixels are unchanged.

        Tesseract is a subprocess call per read; the trigger and Rx regions are
        read on every poll and almost never change between polls.
        """
        pixels = screenshot.crop(region).tobytes()
        cached = self._region_ocr_cache.get(key)
        if cached is not None and cached[0] == pixels:
            return cached[1]
        text = read()
        self._region_ocr_cache[key] = (pixels, text)
        return text

    def _extract_rx_number(self, screenshot: Image.Image) -> str:
        """Extract the Rx number from the rx_number region with validation.

        Always uses OCR (Tesseract preferred, EasyOCR fallback), in both
        verification modes. Returns "" when no trustworthy number was read; a
        plausible-but-odd reading is left in self._unconfirmed_rx for the monitor
        loop to accept once it has held still (see _resolve_unconfirmed_rx).
        """
        self._unconfirmed_rx = ""
        self._rx_verdict = ""
        self._rx_raw_text = ""
        try:
            rx_region = tuple(self.config["regions"].get("rx_number") or self.config["regions"]["trigger"])
            rx_ocr = get_cached_ocr_provider("tesseract", self.advanced_settings)
            recent_rxs = list(self.processed_rx_times.keys())

            rx_text = self._ocr_region_cached(
                "rx", screenshot, rx_region,
                lambda: rx_ocr.get_text_from_region(screenshot, rx_region),
            )
            self._rx_raw_text = rx_text
            rx_number, verdict = trigger_logic.parse_rx_text(rx_text, self.current_session_rx, recent_rxs)

            # Second opinion from the padded digits-only read before giving up.
            if verdict != trigger_logic.RX_OK and hasattr(rx_ocr, "read_digits"):
                digits = self._ocr_region_cached(
                    "rx_digits", screenshot, rx_region,
                    lambda: rx_ocr.read_digits(screenshot, rx_region),
                )
                alt_number, alt_verdict = trigger_logic.parse_rx_text(digits, self.current_session_rx, recent_rxs)
                if alt_verdict == trigger_logic.RX_OK or not rx_number:
                    rx_number, verdict = alt_number, alt_verdict

            self._rx_verdict = verdict
            if verdict == trigger_logic.RX_OK:
                return rx_number
            if verdict == trigger_logic.RX_UNCONFIRMED:
                self._unconfirmed_rx = rx_number
            logging.debug(f"Rx extraction - no valid number in '{rx_text}' (verdict: {verdict or 'none'})")
            return ""
        except Exception as e:
            logging.error(f"Error extracting Rx number: {e}")
            return ""

    def _resolve_unconfirmed_rx(self, now: float) -> str:
        """Accept an odd Rx reading once it has stayed the same long enough.

        A reading whose length does not fit the current Rx is usually a one-poll
        artifact of the screen redrawing. But OCR also misreads real numbers, and
        it misreads the same pixels the same way every time, so rejecting forever
        leaves that prescription unverified. The number is only used to tell
        prescriptions apart -- verification looks at the screen -- so a stable
        wrong reading is still a usable identity.
        """
        held = self._unconfirmed_tracker.observe(self._unconfirmed_rx, now)
        if not self._unconfirmed_rx:
            return ""
        required = float(self.advanced_settings.get("trigger", {}).get("unconfirmed_rx_stable_seconds", 2.0))
        if held < required:
            return ""
        logging.info(
            f"Accepting unconfirmed Rx reading '{self._unconfirmed_rx}' - unchanged for {held:.1f}s "
            f"(current session Rx: {self.current_session_rx})"
        )
        self._unconfirmed_tracker.reset()
        return self._unconfirmed_rx

    def _report_unreadable_rx(self, now: float):
        """Trigger is on screen but no Rx number could be read: say so."""
        if self._rx_verdict == trigger_logic.RX_PARTIAL and self.current_session_rx:
            # Tail of the Rx we just handled; the screen is still redrawing.
            return
        if not self._rx_unreadable_since:
            self._rx_unreadable_since = now
        trigger_config = self.advanced_settings.get("trigger", {})
        warn_after = float(trigger_config.get("rx_unreadable_warn_seconds", 3.0))
        if now - self._rx_unreadable_since < warn_after:
            self._set_state("reading", "Reading Rx number")
            return
        self._set_state("unreadable", "Can't read the Rx number")
        log_interval = float(trigger_config.get("rx_unreadable_log_interval_seconds", 10.0))
        if now - self._rx_unreadable_logged >= log_interval:
            self._rx_unreadable_logged = now
            logging.info(
                f"Trigger detected but no Rx number readable for {now - self._rx_unreadable_since:.1f}s "
                f"(OCR text: '{self._rx_raw_text}')"
            )

    def _check_trigger(self, screenshot: Image.Image) -> Tuple[bool, str]:
        """Check if the trigger text is present.

        Both verification modes use OCR for trigger detection; the VLM is only
        used for field verification.
        """
        trigger_config = self.advanced_settings.get("trigger", {})
        trigger_region = tuple(self.config["regions"]["trigger"])
        keywords = trigger_config.get("keywords", ["pre", "check", "rx"])
        self.trigger_check_count += 1
        return self._check_trigger_with_ocr(screenshot, trigger_region, keywords, trigger_config)

    def _check_trigger_with_ocr(self, screenshot: Image.Image, trigger_region: tuple, keywords: list, trigger_config: dict) -> Tuple[bool, str]:
        """OCR-based trigger detection.

        IMPORTANT: Always attempt Tesseract first for trigger detection, regardless of the
        globally configured OCR provider, to maximize speed and stability. If Tesseract is
        unavailable, get_cached_ocr_provider falls back to EasyOCR.
        """
        try:
            trigger_ocr = get_cached_ocr_provider("tesseract", self.advanced_settings)
            trigger_text = self._ocr_region_cached(
                "trigger", screenshot, trigger_region,
                lambda: trigger_ocr.get_text_from_region(screenshot, trigger_region),
            )

            trigger_detected = trigger_logic.trigger_text_matches(
                trigger_text,
                keywords,
                trigger_config.get("keyword_similarity_threshold", 90),
                trigger_config.get("min_keyword_matches", 2),
            )
            rx_number = self._extract_rx_number(screenshot) if trigger_detected else ""
            return trigger_detected, rx_number

        except Exception as e:
            logging.warning(f"OCR trigger detection failed: {e}")
            return False, ""

    async def _perform_ocr_on_all_fields(
        self, screenshot: Image.Image
    ) -> Dict[str, Tuple[str, str]]:
        """Performs OCR on all configured fields concurrently using a process pool."""
        logging.info("Starting concurrent OCR processing on all fields...")

        screenshot_bytes = screenshot.tobytes()
        width, height = screenshot.size
        tasks = []
        fields_to_process_map: Dict[str, Tuple[str, str]] = {}

        enabled_fields = ["patient_name", "prescriber_name", "drug_name", "direction_sig"]
        enabled_optional_fields = self.config.get("optional_fields_enabled", {})
        enabled_fields.extend([field for field, is_enabled in enabled_optional_fields.items() if is_enabled])

        for field_name in self.config["regions"]["fields"]:
            if field_name not in enabled_fields:
                continue

            config = self.config["regions"]["fields"][field_name]
            for region_type in ["entered", "source"]:
                field_identifier = f"{field_name}_{region_type}"
                region = tuple(config[region_type])
                
                func = functools.partial(
                    perform_ocr_task,
                    self.ocr_provider_type,
                    self.advanced_settings,
                    screenshot_bytes,
                    width,
                    height,
                    region,
                    field_identifier,
                )
                task = self.loop.run_in_executor(self.process_pool, func)
                tasks.append(task)
                fields_to_process_map[field_identifier] = (field_name, region_type)
        
        ocr_results_flat = {}
        results = await asyncio.gather(*tasks)

        for field_identifier, text in results:
            ocr_results_flat[field_identifier] = text

        ocr_results = {}
        for field_name in enabled_fields:
            if field_name in self.config["regions"]["fields"]:
                entered = ocr_results_flat.get(f"{field_name}_entered", "")
                source = ocr_results_flat.get(f"{field_name}_source", "")
                ocr_results[field_name] = (entered, source)
                logging.info(
                    f"Completed OCR for {field_name} | Entered: '{entered[:50]}...' | Source: '{source[:50]}...'"
                )

        logging.info(f"Completed OCR processing for {len(ocr_results)} fields")
        return ocr_results

    async def _verify_all_fields(
        self,
        screenshot: Image.Image,
        ocr_results: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> bool:
        """Run verification on all fields and show the overlay.

        Returns False when no result could be produced (e.g. the AI call failed),
        so the caller can let this prescription be tried again.
        """
        if self.verification_in_progress:
            logging.debug("Verification already in progress, skipping...")
            return True

        rx_label = f"Rx#{self.last_rx_number}" if self.last_rx_number else "Rx"
        try:
            self.verification_in_progress = True
            logging.info("Running field verification...")
            self._set_state("checking", f"Checking {rx_label}")

            verification_method = self._verification_method()

            if verification_method == "vlm_ai":
                # Use VLM verification (direct image analysis)
                results = await self._verify_with_vlm()
            else:
                # Use local OCR + fuzzy matching (default)
                if ocr_results is None:
                    ocr_results = await self._perform_ocr_on_all_fields(screenshot)
                results = self.comparison_engine.verify_fields(ocr_results)

            if not results:
                logging.error(f"Verification produced no result for {rx_label}")
                self._set_state("error", f"{rx_label} — check failed, will retry")
                return False

            log_rx_summary(self.last_rx_number or "", results)

            # Create prescription signature based on method
            if verification_method == "vlm_ai":
                self.last_verified_signature = f"vlm_verification_{int(time.time())}"
            elif ocr_results:
                self.last_verified_signature = self._get_prescription_signature(ocr_results)
            else:
                self.last_verified_signature = f"verification_{int(time.time())}"

            matches = sum(1 for r in results.values() if r["match"])
            if matches == len(results):
                await self._handle_all_fields_matched()
                self._set_state("match", f"{rx_label} — all {matches} fields matched")
            else:
                self._set_state("review", f"{rx_label} — review: {matches} of {len(results)} matched")

            self._show_overlay(results)
            return True
        except Exception as e:
            logging.error(f"Error during verification: {e}")
            self._set_state("error", f"{rx_label} — check failed, will retry")
            return False
        finally:
            self.verification_in_progress = False

    async def _verify_with_vlm(self) -> Dict[str, Dict[str, Any]]:
        """Perform verification using Vision Language Model"""
        try:
            # Get cached VLM verifier (loads and validates the VLM configuration)
            vlm_verifier = self._get_cached_vlm_verifier()
            if not vlm_verifier:
                logging.error("VLM: Failed to get VLM verifier")
                return {}
            
            # Run VLM verification
            logging.info("VLM: Starting vision-based verification")
            vlm_scores = vlm_verifier.verify_with_vlm()

            # No scores means the call failed; all zeros almost always means the
            # source image had not finished loading. Either way, look again.
            retry_attempts = int(vlm_verifier.settings.get("retry_attempts", 1))
            retry_delay = float(vlm_verifier.settings.get("retry_delay_seconds", 1.0))
            for attempt in range(1, retry_attempts + 1):
                if self.should_stop or (vlm_scores and any(vlm_scores.values())):
                    break
                reason = "all scores were 0" if vlm_scores else "the request failed"
                logging.warning(f"VLM: {reason}, retrying in {retry_delay}s (attempt {attempt}/{retry_attempts})")
                self._set_state("checking", f"Rx#{self.last_rx_number} — retrying check")
                await asyncio.sleep(retry_delay)
                vlm_scores = vlm_verifier.verify_with_vlm()
            
            # Convert VLM category scores to field-level results format for overlay
            results = {}
            
            # Get thresholds for comparison
            thresholds = self.config.get("thresholds", {})
            
            # Map VLM category scores to display fields with coordinates
            # VLM returns: {"patient": score, "prescriber": score, "drug": score, "direction": score}
            category_to_field_map = {
                "patient": "patient_name",
                "prescriber": "prescriber_name", 
                "drug": "drug_name",
                "direction": "direction_sig"
            }
            
            # Map VLM categories to threshold keys (some differ from category names)
            category_to_threshold_map = {
                "patient": "patient",
                "prescriber": "prescriber",
                "drug": "drug", 
                "direction": "sig"  # VLM uses "direction" but threshold key is "sig"
            }
            
            for category, score in vlm_scores.items():
                # Get the field name for coordinates lookup
                field_name = category_to_field_map.get(category, category)
                
                # Get threshold for this category using proper threshold key
                threshold_key = category_to_threshold_map.get(category, category)
                threshold = thresholds.get(threshold_key, 70)
                
                match = score >= threshold
                
                # Get coordinates for overlay from OCR field configuration
                field_coords = self.config.get("regions", {}).get("fields", {}).get(field_name, {}).get("entered", [])
                
                if not field_coords or len(field_coords) != 4:
                    logging.debug(f"VLM: No valid coordinates for {field_name}, skipping overlay box")
                    field_coords = []
                
                results[field_name] = {
                    "entered": f"VLM_CATEGORY: {category}",
                    "source": f"VLM_IMAGE_ANALYSIS", 
                    "score": score,
                    "match": match,
                    "threshold": threshold,
                    "method": "vlm_ai",
                    "coords": field_coords  # Add coordinates for overlay (may be empty)
                }
                
                # Removed redundant logging here - will be logged by log_rx_summary
            
            logging.info(f"VLM: Verification completed for {len(results)} fields")
            return results
            
        except Exception as e:
            logging.error(f"VLM: Error during verification: {e}")
            logging.error(f"VLM: Falling back to empty results")
            return {}

    def _load_vlm_config(self) -> Optional[Dict[str, Any]]:
        """Load VLM configuration from config/vlm_config.json with environment variable substitution"""
        try:
            vlm_config_file = os.path.join("config", "vlm_config.json")
            if os.path.exists(vlm_config_file):
                with open(vlm_config_file, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                
                # Substitute environment variables
                config = substitute_env_vars(raw_config)
                logging.debug(f"VLM: Configuration loaded from {vlm_config_file} with environment variable substitution")
                return config
            else:
                logging.error(f"VLM: Configuration file {vlm_config_file} not found")
                return None
        except Exception as e:
            logging.error(f"VLM: Error loading configuration: {e}")
            return None



    def stop(self):
        """Signal the monitoring loop to stop. Non-blocking, safe from any thread.

        The process pool is torn down by async_run's finally clause, so callers
        should join the monitoring thread to know teardown has finished. Doing it
        here instead would block the caller (the UI thread) on in-flight OCR.
        """
        logging.info("Stop requested - monitoring will terminate...")
        self.should_stop = True
        self._close_overlay()

    def _shutdown_pool(self):
        """Shut the OCR process pool down. Idempotent."""
        if getattr(self, "_pool_closed", False):
            return
        self._pool_closed = True
        try:
            self.process_pool.shutdown(wait=True)
        except Exception as e:
            logging.error(f"Error shutting down OCR process pool: {e}")

    async def async_run(self):
        """Main asynchronous monitoring loop."""
        verification_method = self._verification_method()

        trigger_keywords = self.advanced_settings.get("trigger", {}).get("keywords", ["pre", "check", "rx"])
        
        # Display appropriate monitoring message based on method
        method_descriptions = {
            "local_ocr_fuzzy": "📖 Local OCR + Fuzzy matching",
            "vlm_ai": "👁️ VLM AI (Direct image analysis)"
        }
        
        method_desc = method_descriptions.get(verification_method, f"❓ Unknown method ({verification_method})")
        logging.info(f"{method_desc} monitoring active - Looking for triggers: {trigger_keywords}")
        self._waiting_text = f"Waiting for Rx — {method_desc}"
        self._set_state("waiting", self._waiting_text)

        try:
            await self._monitor_loop()
        finally:
            self._close_overlay()
            self._shutdown_pool()
            self._set_state("stopped", "Stopped")
            logging.info("Monitoring loop exited")

    async def _monitor_loop(self):
        """The polling loop itself. Teardown is handled by async_run."""
        consecutive_no_change = 0
        loop_count = 0

        while not self.should_stop:
            try:
                loop_count += 1
                screenshot = pyautogui.screenshot()

                screen_changed = self._has_screen_changed(screenshot)
                if screen_changed:
                    consecutive_no_change = 0
                    if self.overlay_visible and (time.time() - self.overlay_created_time) > self.advanced_settings.get("overlay", {}).get("min_display_seconds", 3.0):
                        self._close_overlay()
                else:
                    consecutive_no_change += 1

                trigger_detected, current_rx_number = self._check_trigger(screenshot)
                now = time.time()
                if trigger_detected:
                    self.last_seen_trigger_time = now
                    if current_rx_number:
                        self._unconfirmed_tracker.reset()
                    else:
                        current_rx_number = self._resolve_unconfirmed_rx(now)
                    if current_rx_number:
                        self._rx_unreadable_since = 0.0
                    logging.debug(f"Trigger detected: Rx#{current_rx_number or 'UNKNOWN'}, recently_triggered={self.recently_triggered}, last_rx={self.last_rx_number}")
                else:
                    self._unconfirmed_tracker.reset()
                    self._rx_unreadable_since = 0.0
                    self._set_state("waiting", self._waiting_text)

                # Process trigger regardless of recently_triggered state
                if trigger_detected:
                    cooldown = float(self.config.get("timing", {}).get("same_prescription_wait_seconds", 3.0))

                    # Never reprocess the same Rx while it stays on screen. Partial
                    # and odd-length readings were already filtered by
                    # _extract_rx_number / _resolve_unconfirmed_rx.
                    should_process = False
                    process_reason = ""
                    skip_reason = ""
                    
                    if not current_rx_number:
                        skip_reason = "no Rx number extracted"
                        self._report_unreadable_rx(now)
                    elif current_rx_number == self.current_session_rx:
                        skip_reason = "same as current session prescription"
                    elif current_rx_number in self.processed_rx_times:
                        # This Rx was processed before - check cooldown regardless of session state
                        time_since_processed = now - self.processed_rx_times[current_rx_number]
                        if time_since_processed < cooldown:
                            skip_reason = f"processed {time_since_processed:.1f}s ago (cooldown: {cooldown - time_since_processed:.1f}s remaining)"
                        else:
                            should_process = True
                            process_reason = f"returning after {time_since_processed:.1f}s"
                    elif self.current_session_rx is None:
                        should_process = True
                        process_reason = "first"
                    else:
                        should_process = True
                        process_reason = "new"
                    
                    if should_process:
                        logging.info(f"Processing Rx#{current_rx_number} ({process_reason}) - State change: recently_triggered={self.recently_triggered} -> True")
                        self._set_state("reading", f"Rx#{current_rx_number} — reading screen")
                        self.last_rx_number = current_rx_number
                        self.current_session_rx = current_rx_number  # Track current session
                        self.recently_triggered = True
                        self.last_trigger_time = now
                        self.processed_rx_times[current_rx_number] = now
                        self.skip_count_for_current_rx = 0  # Reset skip counter for new Rx
                        
                        delay = self.config.get("timing", {}).get("trigger_content_load_delay_seconds", 0.5)
                        await asyncio.sleep(delay)
                        
                        fresh_screenshot = pyautogui.screenshot()
                        if not await self._verify_all_fields(fresh_screenshot):
                            # Leave the session so the cooldown above lets this
                            # prescription be tried again while it stays on screen.
                            self.current_session_rx = None
                    else:
                        # Skip processing with smart logging
                        self.skip_count_for_current_rx += 1
                        
                        # Smart logging: show first skip immediately, then every 10th skip
                        if self.skip_count_for_current_rx == 1 or self.skip_count_for_current_rx % 10 == 0:
                            if current_rx_number:
                                logging.info(f"SKIPPING: Rx#{current_rx_number} - {skip_reason} (skipped {self.skip_count_for_current_rx} times)")
                            else:
                                logging.debug(f"Trigger detected but {skip_reason} (skipped {self.skip_count_for_current_rx} times)")
                        
                # Handle reset when trigger is absent (only when no trigger is currently detected)
                elif self.recently_triggered:
                    reset_delay = self.advanced_settings.get("trigger", {}).get("lost_reset_delay_seconds", 5.0)
                    time_since_last_trigger = now - self.last_seen_trigger_time
                    if time_since_last_trigger > reset_delay:
                        logging.info(f"Trigger text absent for {time_since_last_trigger:.1f}s (>{reset_delay}s), resetting for next prescription - State change: recently_triggered=True -> False")
                        self.recently_triggered = False
                        self.last_rx_number = None
                        self.current_session_rx = None  # Clear current session
                        self.skip_count_for_current_rx = 0
                        
                        # More aggressive cleanup: remove entries older than 5 minutes
                        cleanup_threshold = now - 300  # 5 minutes
                        old_entries = [rx for rx, timestamp in self.processed_rx_times.items() if timestamp < cleanup_threshold]
                        for rx in old_entries:
                            del self.processed_rx_times[rx]
                        if old_entries:
                            logging.debug(f"Cleaned up {len(old_entries)} old Rx entries from memory")
                        
                        self._close_overlay()
                    else:
                        # Smart logging: only log every 20 checks when waiting for reset
                        if self.trigger_check_count % 20 == 0:
                            logging.debug(f"Waiting for trigger reset: {time_since_last_trigger:.1f}s/{reset_delay}s elapsed")
                
                await asyncio.sleep(self.config["timing"]["fast_polling_seconds"])

            except asyncio.CancelledError:
                logging.info("Main loop cancelled.")
                break
            except Exception as e:
                logging.error(f"Error in main async loop: {e}", exc_info=True)
                await asyncio.sleep(1)


def load_config(path: str) -> Optional[Dict[str, Any]]:
    """Loads configuration from a JSON file."""
    try:
        with open(path, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {path}")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from configuration file: {path}")
    return None


def main():
    """Application entry point."""
    setup_logging()
    config = load_config("config/config.json")
    if not config:
        logging.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    controller = VerificationController(config, loop)

    try:
        logging.info("Starting application event loop...")
        loop.run_until_complete(controller.async_run())
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Shutting down...")
    finally:
        controller.stop()
        # Clean up any remaining tasks
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        
        # Gather and wait for all tasks to be cancelled
        async def gather_cancelled():
            await asyncio.gather(*tasks, return_exceptions=True)

        # Run the cleanup gathering
        loop.run_until_complete(gather_cancelled())
        loop.close()
        logging.info("Application shut down gracefully.")


if __name__ == "__main__":
    main()
