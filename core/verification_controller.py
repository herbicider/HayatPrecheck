
import asyncio
import functools
import json
import logging
import re
import sys
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Optional, Tuple

import pyautogui
import tkinter as tk
from PIL import Image, ImageFilter
from dotenv import load_dotenv

from core.comparison_engine import ComparisonEngine
from core.logger_config import log_rx_summary, setup_logging
from core.ocr_provider import get_cached_ocr_provider


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


def substitute_env_vars(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively substitute environment variables in configuration values.
    Supports ${VAR_NAME} syntax.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    def substitute_string(value: str) -> str:
        """Substitute environment variables in a string"""
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_var, value)
    
    def substitute_recursive(obj):
        """Recursively process the configuration object"""
        if isinstance(obj, dict):
            return {key: substitute_recursive(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [substitute_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return substitute_string(obj)
        else:
            return obj
    
    return substitute_recursive(config_dict)


class VerificationController:
    """Manages the main async application loop, screen monitoring, and UI."""

    def __init__(self, config: Dict[str, Any], loop: asyncio.AbstractEventLoop):
        self.config = config
        self.loop = loop
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
        self.overlay_root = None
        
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

        # Optional OCR warm-up to avoid first-use latency
        try:
            warm_up_main = bool(startup_cfg.get("warm_up_ocr_on_start", False))
            warm_up_workers = bool(startup_cfg.get("warm_up_ocr_workers", False))
            workers_to_warm = int(startup_cfg.get("workers_to_warm", 2))

            if warm_up_main:
                self._warm_up_main_ocr()
            if warm_up_workers:
                self._warm_up_worker_processes(max(1, workers_to_warm))
        except Exception as e:
            logging.debug(f"OCR warm-up skipped due to error: {e}")

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

    def _show_tk_overlay(self, results: Dict[str, Any]):
        """Displays a transparent overlay with colored rectangles."""
        def _task():
            try:
                if self.overlay_root:
                    self.overlay_root.destroy()
                
                root = tk.Tk()
                self.overlay_root = root
                self.overlay_created_time = time.time()
                root.overrideredirect(True)
                root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
                root.lift()
                root.wm_attributes("-topmost", True)
                root.wm_attributes("-disabled", True)
                root.wm_attributes("-transparentcolor", "white")

                canvas = tk.Canvas(root, bg='white', highlightthickness=0)
                canvas.pack(fill="both", expand=True)

                for result in results.values():
                    # Check if coords exist before trying to draw
                    if "coords" in result and len(result["coords"]) == 4:
                        # Get score and threshold for gradient coloring
                        score = result.get("score", 0)
                        threshold = result.get("threshold", 80)
                        
                        # Color scheme based on score:
                        # 100 = Dark green (#006400)
                        # 100-threshold = Light green (#90EE90)
                        # threshold-1 = Light red (#FFB6C1) 
                        # 0 = Red (#FF0000)
                        
                        if score == 100:
                            color = "#006400"  # Dark green
                        elif score >= threshold:
                            # Light green gradient from threshold to 100
                            # Interpolate between light green and dark green
                            ratio = (score - threshold) / (100 - threshold) if threshold < 100 else 1
                            # Light green RGB(144,238,144) to Dark green RGB(0,100,0)
                            r = int(144 * (1 - ratio))
                            g = int(238 * (1 - ratio) + 100 * ratio)
                            b = int(144 * (1 - ratio))
                            color = f"#{r:02x}{g:02x}{b:02x}"
                        elif score > 0:
                            # Light red gradient from 1 to threshold-1
                            # Interpolate between red and light red
                            ratio = score / threshold if threshold > 0 else 0
                            # Red RGB(255,0,0) to Light red RGB(255,182,193)
                            r = 255
                            g = int(182 * ratio)
                            b = int(193 * ratio)
                            color = f"#{r:02x}{g:02x}{b:02x}"
                        else:
                            color = "#FF0000"  # Red for score 0
                        
                        canvas.create_rectangle(*result["coords"], outline=color, width=3)
                    else:
                        logging.warning(f"Skipping overlay for field - missing or invalid coords: {result.get('coords', 'None')}")

                root.update()
                logging.info("Overlay displayed successfully.")
            except Exception as e:
                logging.error(f"Failed to create Tkinter overlay: {e}")
        
        # Tkinter calls should be made from the main thread.
        self.loop.call_soon_threadsafe(_task)


    def _close_overlay(self):
        """Close the current overlay if it exists."""
        def _task():
            if self.overlay_root:
                try:
                    self.overlay_root.destroy()
                    self.overlay_root = None
                except tk.TclError:
                    self.overlay_root = None
        
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(_task)

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

        key_to_send = automation_config.get("key_on_all_match", "f12")
        delay_seconds = automation_config.get("key_delay_seconds", 0.5)

        logging.info(
            f"SUCCESS: All fields matched! Sending '{key_to_send}' key press in {delay_seconds}s..."
        )
        await asyncio.sleep(delay_seconds)

        try:
            # pyautogui is blocking, but short. For true async, this would also go in an executor.
            pyautogui.press(key_to_send.lower())
            logging.info(f"SUCCESS: Sent '{key_to_send}' key press successfully")
        except Exception as e:
            logging.error(f"Error sending key press: {e}")

    def _is_partial_rx_read(self, rx_number: str, reference_rx: str) -> bool:
        """Check if rx_number is likely a partial read of reference_rx due to UI shifts."""
        if not rx_number or not reference_rx:
            return False
        
        # Check if it's shorter and ends the reference Rx
        if (len(rx_number) < len(reference_rx) and 
            reference_rx.endswith(rx_number) and 
            len(rx_number) >= 3):  # At least 3 digits to be considered a partial match
            return True
            
        return False
    
    def _extract_rx_number(self, screenshot: Image.Image) -> str:
        """Extract the Rx number from the rx_number region with validation.

        Always use OCR (Tesseract preferred, EasyOCR fallback) for Rx number extraction.
        This ensures consistency with the OCR-only trigger detection approach.
        """
        try:
            rx_region = self.config["regions"].get("rx_number") or self.config["regions"]["trigger"]

            from core.ocr_provider import get_cached_ocr_provider
            
            # Try Tesseract first for speed and consistency
            try:
                rx_ocr = get_cached_ocr_provider("tesseract", self.advanced_settings)
                rx_text = rx_ocr.get_text_from_region(screenshot, tuple(rx_region))
                ocr_method = "Tesseract"
            except Exception as tesseract_error:
                logging.debug(f"Tesseract failed for Rx extraction, trying EasyOCR: {tesseract_error}")
                # Fallback to EasyOCR
                rx_ocr = get_cached_ocr_provider("easyocr", self.advanced_settings)
                rx_text = rx_ocr.get_text_from_region(screenshot, tuple(rx_region))
                ocr_method = "EasyOCR"
            
            patterns = [r"rx\s*-\s*(\d+)", r"rx\s+(\d+)", r"(\d{4,})"]
            text_lower = rx_text.lower()
            
            logging.debug(f"Rx extraction ({ocr_method}) - OCR text: '{rx_text}' -> '{text_lower}'")
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, text_lower)
                if match:
                    rx_number = match.group(1)
                    
                    # Validate: must be all digits
                    if not rx_number.isdigit():
                        logging.debug(f"Rx extraction - Pattern {i+1} matched '{rx_number}' but contains non-digits, skipping")
                        continue
                    
                    # Validate: reject partial Rx numbers that are likely UI shift artifacts
                    # Check against current session Rx
                    if self.current_session_rx and self.current_session_rx.isdigit():
                        if self._is_partial_rx_read(rx_number, self.current_session_rx):
                            logging.warning(f"Rx extraction - Rejecting '{rx_number}' as partial read of current session Rx '{self.current_session_rx}' (UI shift artifact)")
                            continue
                        elif len(rx_number) != len(self.current_session_rx):
                            logging.warning(f"Rx extraction - Rejecting '{rx_number}' - length ({len(rx_number)}) differs from current session Rx '{self.current_session_rx}' length ({len(self.current_session_rx)}) (UI shift artifact)")
                            continue
                    
                    # Also check against recently processed Rx numbers
                    is_partial_of_recent = False
                    for recent_rx in self.processed_rx_times.keys():
                        if recent_rx and self._is_partial_rx_read(rx_number, recent_rx):
                            logging.warning(f"Rx extraction - Rejecting '{rx_number}' as partial read of recently processed Rx '{recent_rx}' (UI shift artifact)")
                            is_partial_of_recent = True
                            break
                        elif recent_rx and len(rx_number) != len(recent_rx) and len(rx_number) < len(recent_rx):
                            # Different length and shorter - likely partial read
                            logging.warning(f"Rx extraction - Rejecting '{rx_number}' - suspiciously shorter than recent Rx '{recent_rx}' (likely UI shift artifact)")
                            is_partial_of_recent = True
                            break
                    
                    if is_partial_of_recent:
                        continue
                    
                    logging.debug(f"Rx extraction ({ocr_method}) - Pattern {i+1} matched and validated: '{rx_number}'")
                    return rx_number
            
            logging.debug(f"Rx extraction ({ocr_method}) - No valid patterns matched in: '{text_lower}'")
            return ""
        except Exception as e:
            logging.error(f"Error extracting Rx number: {e}")
            return ""

    def _check_trigger(self, screenshot: Image.Image) -> Tuple[bool, str]:
        """Check if the trigger text is present using OCR-only approach.
        
        Both OCR and VLM verification modes use OCR for trigger detection.
        VLM is only used for field verification, not trigger detection.
        """
        trigger_config = self.advanced_settings.get("trigger", {})
        trigger_region = tuple(self.config["regions"]["trigger"])
        keywords = trigger_config.get("keywords", ["pre", "check", "rx"])
        
        # Increment trigger check counter for smart logging
        self.trigger_check_count += 1
        
        # Always use OCR-based trigger detection first (Tesseract preferred, EasyOCR fallback)
        ocr_trigger_detected, ocr_rx_number = self._check_trigger_with_ocr(screenshot, trigger_region, keywords, trigger_config)
        
        # If OCR succeeded, use it regardless of verification mode
        if ocr_trigger_detected:
            if ocr_rx_number:
                logging.debug(f"Trigger detected via OCR: Rx#{ocr_rx_number}")
            else:
                logging.debug("Trigger detected via OCR (no Rx number)")
            return ocr_trigger_detected, ocr_rx_number
        
        # If in VLM mode and OCR failed completely, try the VLM mode OCR approach
        # (This provides a secondary OCR attempt with potentially different settings)
        verification_method = self.config.get("verification_method", "local_ocr_fuzzy")
        
        # Handle legacy configuration for trigger detection
        if "verification_mode" in self.config and "verification_method" not in self.config:
            legacy_mode = self.config.get("verification_mode", "ocr")
            verification_method = "vlm_ai" if legacy_mode == "vlm" else "local_ocr_fuzzy"
        
        if verification_method == "vlm_ai":
            vlm_trigger_detected, vlm_rx_number = self._check_trigger_with_vlm(trigger_region, keywords)
            if vlm_trigger_detected:
                logging.debug(f"Trigger detected via VLM mode OCR: Rx#{vlm_rx_number or 'UNKNOWN'}")
                return vlm_trigger_detected, vlm_rx_number
            else:
                # Smart logging: only log every 30 checks when no trigger found
                if self.trigger_check_count % 30 == 0:
                    logging.debug(f"VLM mode OCR monitoring - {self.trigger_check_count} checks completed, no triggers detected")
        
        # No trigger detected by any OCR method
        return False, ""
    
    def _check_trigger_with_ocr(self, screenshot: Image.Image, trigger_region: tuple, keywords: list, trigger_config: dict) -> Tuple[bool, str]:
        """Traditional OCR-based trigger detection.

        IMPORTANT: Always attempt Tesseract first for trigger detection, regardless of the
        globally configured OCR provider, to maximize speed and stability. If Tesseract is
        unavailable, gracefully fall back to EasyOCR via the provider cache. This ensures
        the sequence: Tesseract -> EasyOCR -> (only then) VLM fallback handled by caller.
        """
        try:
            # Force Tesseract-first provider selection for trigger text
            # If Tesseract is not available, get_cached_ocr_provider will smart-fallback to EasyOCR
            from core.ocr_provider import get_cached_ocr_provider
            trigger_ocr = get_cached_ocr_provider("tesseract", self.advanced_settings)

            trigger_text = trigger_ocr.get_text_from_region(screenshot, trigger_region)
            
            sim_threshold = trigger_config.get("keyword_similarity_threshold", 90)
            min_matches = trigger_config.get("min_keyword_matches", 2)
            
            from rapidfuzz import fuzz
            text_lower = trigger_text.lower()
            
            # Check for full phrase match first
            full_phrase = " ".join(keywords)
            if fuzz.ratio(text_lower.strip(), full_phrase.lower()) >= 80:
                trigger_detected = True
                rx_number = self._extract_rx_number(screenshot)
                return trigger_detected, rx_number
            
            # Check individual keyword matches
            text_words = re.split(r'[\s\-_.,;:|"\']+', text_lower)
            text_words = [w for w in text_words if w]
            found_count = sum(1 for kw in keywords if any(fuzz.ratio(w, kw.lower()) >= sim_threshold for w in text_words))
            
            trigger_detected = found_count >= min_matches
            rx_number = self._extract_rx_number(screenshot) if trigger_detected else ""
            
            if trigger_detected:
                logging.debug(f"OCR trigger detected - found {found_count}/{len(keywords)} keywords")
            
            return trigger_detected, rx_number
            
        except Exception as e:
            logging.warning(f"OCR trigger detection failed: {e}")
            return False, ""
    
    def _check_trigger_with_vlm(self, trigger_region: tuple, keywords: list) -> Tuple[bool, str]:
        """OCR-only trigger detection for VLM mode (no VLM fallback)
        
        When in VLM mode, we still use OCR for trigger detection per user requirements.
        VLM is only used for field verification, not trigger detection.
        """
        try:
            # Load VLM configuration
            vlm_config = self._load_vlm_config()
            if not vlm_config:
                logging.warning("VLM trigger: No VLM configuration available, using direct OCR")
                # Take a fresh screenshot for OCR trigger detection
                screenshot = pyautogui.screenshot()
                return self._check_trigger_with_ocr(screenshot, trigger_region, keywords, self.advanced_settings.get("trigger", {}))
            
            # Get cached VLM verifier and use OCR trigger detection
            vlm_verifier = self._get_cached_vlm_verifier()
            if not vlm_verifier:
                logging.warning("VLM trigger: Failed to get VLM verifier, using direct OCR")
                screenshot = pyautogui.screenshot()
                return self._check_trigger_with_ocr(screenshot, trigger_region, keywords, self.advanced_settings.get("trigger", {}))
            
            trigger_detected, rx_number = vlm_verifier.detect_trigger_with_ocr(trigger_region, keywords)
            
            if trigger_detected:
                logging.debug(f"VLM mode OCR trigger detection: trigger={trigger_detected}, rx={rx_number}")
            else:
                logging.debug("VLM mode OCR trigger detection: No trigger detected")
            
            return trigger_detected, rx_number
            
        except Exception as e:
            logging.error(f"VLM mode OCR trigger detection failed: {e}")
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
    ):
        """Run verification on all fields and show overlay."""
        if self.verification_in_progress:
            logging.debug("Verification already in progress, skipping...")
            return

        try:
            self.verification_in_progress = True
            logging.info("Running field verification...")

            # Check verification method (new structure)
            verification_method = self.config.get("verification_method", "local_ocr_fuzzy")
            
            # Handle legacy configuration
            if "verification_mode" in self.config and "verification_method" not in self.config:
                legacy_mode = self.config.get("verification_mode", "ocr")
                verification_method = "vlm_ai" if legacy_mode == "vlm" else "local_ocr_fuzzy"
                logging.info(f"Using legacy verification_mode '{legacy_mode}' -> '{verification_method}'")
            
            if verification_method == "vlm_ai":
                # Use VLM verification (direct image analysis)
                results = await self._verify_with_vlm()
            else:
                # Use local OCR + fuzzy matching (default)
                if ocr_results is None:
                    ocr_results = await self._perform_ocr_on_all_fields(screenshot)
                results = self.comparison_engine.verify_fields(ocr_results)
            
            log_rx_summary(self.last_rx_number or "", results)
            
            # Create prescription signature based on method
            if verification_method == "vlm_ai":
                self.last_verified_signature = f"vlm_verification_{int(time.time())}"
            elif ocr_results:
                self.last_verified_signature = self._get_prescription_signature(ocr_results)
            else:
                self.last_verified_signature = f"verification_{int(time.time())}"

            matches = sum(1 for r in results.values() if r["match"])
            if matches > 0 and matches == len(results):
                await self._handle_all_fields_matched()

            self._show_tk_overlay(results)
        except Exception as e:
            logging.error(f"Error during verification: {e}")
        finally:
            self.verification_in_progress = False

    async def _verify_with_vlm(self) -> Dict[str, Dict[str, Any]]:
        """Perform verification using Vision Language Model"""
        try:
            # Load VLM configuration
            vlm_config = self._load_vlm_config()
            if not vlm_config:
                logging.error("VLM: Configuration not found, falling back to empty results")
                return {}
            
            # Get cached VLM verifier
            vlm_verifier = self._get_cached_vlm_verifier()
            if not vlm_verifier:
                logging.error("VLM: Failed to get VLM verifier")
                return {}
            
            # Run VLM verification
            logging.info("VLM: Starting vision-based verification")
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
        """Stop the monitoring loop gracefully."""
        logging.info("Stop requested - monitoring will terminate...")
        self.should_stop = True
        self._close_overlay()
        self.process_pool.shutdown(wait=True)

    async def async_run(self):
        """Main asynchronous monitoring loop."""
        verification_method = self.config.get("verification_method", "local_ocr_fuzzy")
        
        # Handle legacy configuration
        if "verification_mode" in self.config and "verification_method" not in self.config:
            legacy_mode = self.config.get("verification_mode", "ocr")
            verification_method = "vlm_ai" if legacy_mode == "vlm" else "local_ocr_fuzzy"
        
        trigger_keywords = self.advanced_settings.get("trigger", {}).get("keywords", ["pre", "check", "rx"])
        
        # Display appropriate monitoring message based on method
        method_descriptions = {
            "local_ocr_fuzzy": "📖 Local OCR + Fuzzy matching",
            "vlm_ai": "👁️ VLM AI (Direct image analysis)"
        }
        
        method_desc = method_descriptions.get(verification_method, f"❓ Unknown method ({verification_method})")
        logging.info(f"{method_desc} monitoring active - Looking for triggers: {trigger_keywords}")
            
        consecutive_no_change = 0
        loop_count = 0

        while not self.should_stop:
            try:
                loop_count += 1
                screenshot = pyautogui.screenshot()

                screen_changed = self._has_screen_changed(screenshot)
                if screen_changed:
                    consecutive_no_change = 0
                    if self.overlay_root and (time.time() - self.overlay_created_time) > self.advanced_settings.get("overlay", {}).get("min_display_seconds", 3.0):
                        self._close_overlay()
                else:
                    consecutive_no_change += 1

                trigger_detected, current_rx_number = self._check_trigger(screenshot)
                now = time.time()
                if trigger_detected:
                    self.last_seen_trigger_time = now
                    # Enhanced debug logging for trigger state tracking
                    logging.debug(f"Trigger detected: Rx#{current_rx_number or 'UNKNOWN'}, recently_triggered={self.recently_triggered}, last_rx={self.last_rx_number}")

                # Process trigger regardless of recently_triggered state
                if trigger_detected:
                    cooldown = float(self.config.get("timing", {}).get("same_prescription_wait_seconds", 3.0))
                    last_time = self.processed_rx_times.get(current_rx_number, 0)

                    # Robust Rx processing logic - never reprocess the same Rx in the same session
                    should_process = False
                    process_reason = ""
                    skip_reason = ""
                    
                    if not current_rx_number:
                        skip_reason = "no Rx number extracted"
                    elif current_rx_number == self.current_session_rx:
                        # Same Rx as current session - never reprocess during continuous presence
                        skip_reason = "same as current session prescription"
                    elif (self.current_session_rx and 
                          self._is_partial_rx_read(current_rx_number, self.current_session_rx)):
                        # This looks like a partial read of current session Rx (UI shift artifact)
                        skip_reason = f"partial read of current session Rx '{self.current_session_rx}' (UI shift artifact)"
                    elif (self.current_session_rx and 
                          len(current_rx_number) != len(self.current_session_rx) and 
                          len(current_rx_number) < len(self.current_session_rx)):
                        # Different length and shorter than current session - likely UI shift
                        skip_reason = f"shorter than current session Rx '{self.current_session_rx}' (likely UI shift artifact)"
                    elif any(self._is_partial_rx_read(current_rx_number, processed_rx) 
                            for processed_rx in self.processed_rx_times.keys() 
                            if processed_rx and len(processed_rx) > len(current_rx_number)):
                        # This looks like a partial read of a recently processed Rx
                        matching_rx = next((rx for rx in self.processed_rx_times.keys() 
                                          if rx and self._is_partial_rx_read(current_rx_number, rx)), "unknown")
                        skip_reason = f"partial read of recently processed Rx '{matching_rx}' (UI shift artifact)"
                    elif current_rx_number in self.processed_rx_times:
                        # This Rx was processed before - check cooldown regardless of session state
                        time_since_processed = now - self.processed_rx_times[current_rx_number]
                        if time_since_processed < cooldown:
                            skip_reason = f"processed {time_since_processed:.1f}s ago (cooldown: {cooldown - time_since_processed:.1f}s remaining)"
                        else:
                            should_process = True
                            process_reason = f"returning after {time_since_processed:.1f}s"
                    elif self.current_session_rx is None:
                        # First Rx we've seen in this session AND never processed before
                        should_process = True
                        process_reason = "first"
                    elif current_rx_number != self.current_session_rx:
                        # Different Rx from current session AND never processed before  
                        should_process = True
                        process_reason = "new"
                    
                    if should_process:
                        logging.info(f"Processing Rx#{current_rx_number} ({process_reason}) - State change: recently_triggered={self.recently_triggered} -> True")
                        self.last_rx_number = current_rx_number
                        self.current_session_rx = current_rx_number  # Track current session
                        self.recently_triggered = True
                        self.last_trigger_time = now
                        self.processed_rx_times[current_rx_number] = now
                        self.skip_count_for_current_rx = 0  # Reset skip counter for new Rx
                        
                        delay = self.config.get("timing", {}).get("trigger_content_load_delay_seconds", 0.5)
                        await asyncio.sleep(delay)
                        
                        fresh_screenshot = pyautogui.screenshot()
                        await self._verify_all_fields(fresh_screenshot)
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
        with open(path, "r") as f:
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

    loop = asyncio.get_event_loop()
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
