
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

from comparison_engine import ComparisonEngine
from logger_config import log_rx_summary, setup_logging
from ocr_provider import get_cached_ocr_provider


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

        # Simplified synchronous retry logic for the isolated process
        retry_config = advanced_settings.get("timing", {})
        retry_delay = retry_config.get("ocr_retry_delay_seconds", 0.5)
        max_retries = int(retry_config.get("ocr_max_retries", 3))

        for attempt in range(max_retries):
            text = ocr_provider.get_text_from_region(screenshot, region, field_identifier)
            if text and text.strip():
                if attempt > 0:
                    logging.info(
                        f"OCR (process) success for {field_identifier} on attempt {attempt + 1}"
                    )
                return field_identifier, text
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

        logging.error(
            f"OCR (process) failed for {field_identifier} after {max_retries} attempts."
        )
        return field_identifier, ""

    except Exception as e:
        logging.error(f"Error in OCR process task for {field_identifier}: {e}")
        return field_identifier, ""


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
        self.process_pool = ProcessPoolExecutor()

        self.recently_triggered = False
        self.last_rx_number = None
        self.last_screenshot_hash = None
        self.verification_in_progress = False
        self.overlay_root = None
        self.last_trigger_time = 0
        self.last_seen_trigger_time = 0.0
        self.processed_rx_times: Dict[str, float] = {}
        self.last_verified_signature = ""
        self.overlay_created_time = 0
        self.should_stop = False

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
                    color = "#00ff00" if result["match"] else "#ff0000"
                    canvas.create_rectangle(*result["coords"], outline=color, width=3)

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

    def _extract_rx_number(self, screenshot: Image.Image) -> str:
        """Extract the Rx number from the rx_number region."""
        try:
            rx_region = self.config["regions"].get("rx_number") or self.config["regions"]["trigger"]
            rx_text = self.ocr_provider.get_text_from_region(screenshot, tuple(rx_region))
            
            patterns = [r"rx\s*-\s*(\d+)", r"rx\s+(\d+)", r"(\d{4,})"]
            text_lower = rx_text.lower()
            
            logging.debug(f"Rx extraction - OCR text: '{rx_text}' -> '{text_lower}'")
            
            for i, pattern in enumerate(patterns):
                match = re.search(pattern, text_lower)
                if match:
                    rx_number = match.group(1)
                    logging.debug(f"Rx extraction - Pattern {i+1} matched: '{rx_number}'")
                    return rx_number
            
            logging.debug(f"Rx extraction - No patterns matched in: '{text_lower}'")
            return ""
        except Exception as e:
            logging.error(f"Error extracting Rx number: {e}")
            return ""

    def _check_trigger(self, screenshot: Image.Image) -> Tuple[bool, str]:
        """Check if the trigger text is present."""
        trigger_config = self.advanced_settings.get("trigger", {})
        trigger_region = tuple(self.config["regions"]["trigger"])
        trigger_text = self.ocr_provider.get_text_from_region(screenshot, trigger_region)
        
        keywords = trigger_config.get("keywords", ["pre", "check", "rx"])
        sim_threshold = trigger_config.get("keyword_similarity_threshold", 90)
        min_matches = trigger_config.get("min_keyword_matches", 2)
        
        from rapidfuzz import fuzz
        text_lower = trigger_text.lower()
        
        full_phrase = " ".join(keywords)
        if fuzz.ratio(text_lower.strip(), full_phrase.lower()) >= 80:
             trigger_detected = True
             rx_number = self._extract_rx_number(screenshot)
             return trigger_detected, rx_number

        text_words = re.split(r'[\s\-_.,;:|"\']+', text_lower)
        text_words = [w for w in text_words if w]
        found_count = sum(1 for kw in keywords if any(fuzz.ratio(w, kw.lower()) >= sim_threshold for w in text_words))
        
        trigger_detected = found_count >= min_matches
        rx_number = self._extract_rx_number(screenshot) if trigger_detected else ""
        
        if trigger_detected:
            logging.debug(f"Trigger detected - Rx#{rx_number or 'UNKNOWN'}")
        
        return trigger_detected, rx_number

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

            if ocr_results is None:
                ocr_results = await self._perform_ocr_on_all_fields(screenshot)

            results = self.comparison_engine.verify_fields(ocr_results)
            log_rx_summary(self.last_rx_number or "", results)
            self.last_verified_signature = self._get_prescription_signature(ocr_results)

            matches = sum(1 for r in results.values() if r["match"])
            if matches > 0 and matches == len(results):
                await self._handle_all_fields_matched()

            self._show_tk_overlay(results)
        except Exception as e:
            logging.error(f"Error during verification: {e}")
        finally:
            self.verification_in_progress = False

    def stop(self):
        """Stop the monitoring loop gracefully."""
        logging.info("Stop requested - monitoring will terminate...")
        self.should_stop = True
        self._close_overlay()
        self.process_pool.shutdown(wait=True)

    async def async_run(self):
        """Main asynchronous monitoring loop."""
        logging.info("Starting to monitor for 'pre-check rx' text...")
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

                if trigger_detected:
                    cooldown = float(self.config.get("timing", {}).get("same_prescription_wait_seconds", 3.0))
                    last_time = self.processed_rx_times.get(current_rx_number, 0)

                    # Check if this is a new Rx number (different from the last processed one)
                    is_new_rx = current_rx_number and current_rx_number != self.last_rx_number
                    
                    # Check if enough time has passed since we last processed this specific Rx number
                    is_off_cooldown = (now - last_time) >= cooldown
                    
                    # For the very first Rx (when last_rx_number is None), we should process it
                    is_first_rx = self.last_rx_number is None and current_rx_number
                    
                    # Only process if it's a new Rx number OR it's the first Rx we've seen
                    # Remove the recently_triggered check since we're now tracking by Rx number
                    if is_new_rx or is_first_rx:
                        logging.info(f"Processing Rx#{current_rx_number} ({'first' if is_first_rx else 'new'})")
                        self.last_rx_number = current_rx_number
                        self.recently_triggered = True
                        self.last_trigger_time = now
                        if current_rx_number:
                            self.processed_rx_times[current_rx_number] = now
                        
                        delay = self.config.get("timing", {}).get("trigger_content_load_delay_seconds", 0.5)
                        await asyncio.sleep(delay)
                        
                        fresh_screenshot = pyautogui.screenshot()
                        await self._verify_all_fields(fresh_screenshot)
                    elif current_rx_number == self.last_rx_number:
                        # Same Rx number as before, skip processing but log it
                        logging.debug(f"Skipping Rx#{current_rx_number} - already processed (same as current)")
                    elif not is_off_cooldown:
                        # Different Rx but still in cooldown period
                        logging.debug(f"Skipping Rx#{current_rx_number} - cooldown period ({cooldown}s) not elapsed")
                    elif not current_rx_number:
                        # Trigger detected but no Rx number found (OCR issue)
                        logging.debug(f"Trigger detected but no Rx number extracted - skipping")
                    else:
                        # This shouldn't happen, but log it for debugging
                        logging.debug(f"Trigger detected for Rx#{current_rx_number} but conditions not met for processing")

                elif self.recently_triggered:
                    reset_delay = self.advanced_settings.get("trigger", {}).get("lost_reset_delay_seconds", 5.0)
                    if (now - self.last_seen_trigger_time) > reset_delay:
                        logging.info("Trigger text absent, resetting for next prescription")
                        self.recently_triggered = False
                        self.last_rx_number = None
                        self._close_overlay()
                
                sleep_time = self.config["timing"]["fast_polling_seconds"] if consecutive_no_change < 10 else self.config["timing"]["max_static_sleep_seconds"]
                await asyncio.sleep(sleep_time)

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
    config = load_config("config.json")
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
