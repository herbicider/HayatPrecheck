
import pyautogui
import tkinter as tk
import time
import os
import sys
import json
import logging
import re
from typing import Dict, Any, Tuple, Optional
from PIL import Image, ImageFilter

from ocr_provider import TesseractOcrProvider
from comparison_engine import ComparisonEngine
from logger_config import setup_logging, log_rx_summary

class VerificationController:
    """Manages the main application loop, screen monitoring, and UI."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.advanced_settings = config.get("advanced_settings", {})
        
        self.ocr_provider = TesseractOcrProvider(self.advanced_settings)
        self.comparison_engine = ComparisonEngine(config)

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
            for field_name in ["patient_name", "drug_name"]:
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

    def _close_overlay(self):
        """Close the current overlay if it exists."""
        if self.overlay_root:
            try:
                self.overlay_root.destroy()
                self.overlay_root = None
            except tk.TclError:
                self.overlay_root = None

    def _handle_all_fields_matched(self):
        """Handle when all fields match."""
        automation_config = self.config.get("automation", {})
        if not automation_config.get("send_key_on_all_match"):
            return
            
        key_to_send = automation_config.get("key_on_all_match", "f12")
        delay_seconds = automation_config.get("key_delay_seconds", 0.5)
        
        logging.info(f"SUCCESS: All fields matched! Sending '{key_to_send}' key press in {delay_seconds}s...")
        time.sleep(delay_seconds)
        
        try:
            pyautogui.press(key_to_send.lower())
            logging.info(f"SUCCESS: Sent '{key_to_send}' key press successfully")
        except Exception as e:
            logging.error(f"Error sending key press: {e}")

    def _extract_rx_number(self, trigger_text: str) -> str:
        """Extract the Rx number from the trigger text."""
        patterns = [r'rx\s*-\s*(\d+)', r'rx\s+(\d+)', r'(\d{4,})']
        text_lower = trigger_text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return match.group(1)
        return ""

    def _check_trigger(self, screenshot: Image.Image) -> Tuple[bool, str]:
        """Check if the trigger text is present."""
        trigger_config = self.advanced_settings.get("trigger", {})
        trigger_region = tuple(self.config["regions"]["trigger"])
        trigger_text = self.ocr_provider.get_text_from_region(screenshot, trigger_region)
        
        keywords = trigger_config.get("keywords", ["pre", "check", "rx"])
        sim_threshold = trigger_config.get("keyword_similarity_threshold", 70)
        min_matches = trigger_config.get("min_keyword_matches", 2)
        
        from rapidfuzz import fuzz
        text_lower_words = trigger_text.lower().split()
        found_count = sum(1 for kw in keywords if any(fuzz.ratio(w, kw) >= sim_threshold for w in text_lower_words))
        
        trigger_detected = found_count >= min_matches
        rx_number = self._extract_rx_number(trigger_text) if trigger_detected else ""
        
        return trigger_detected, rx_number
    
    def _perform_ocr_on_all_fields(self, screenshot: Image.Image) -> Dict[str, Tuple[str, str]]:
        """Performs OCR on all configured fields and returns the raw text."""
        ocr_results = {}
        fields_config = self.config["regions"]["fields"]
        for field_name, config in fields_config.items():
            entered_text = self.ocr_provider.get_text_from_region(screenshot, tuple(config["entered"]), f"{field_name}_entered")
            source_text = self.ocr_provider.get_text_from_region(screenshot, tuple(config["source"]), f"{field_name}_source")
            ocr_results[field_name] = (entered_text, source_text)
        return ocr_results

    def _verify_all_fields(self, screenshot: Image.Image):
        """Run verification on all fields and show overlay."""
        if self.verification_in_progress:
            logging.debug("Verification already in progress, skipping...")
            return
            
        try:
            self.verification_in_progress = True
            logging.info("Running field verification...")
            
            ocr_results = self._perform_ocr_on_all_fields(screenshot)
            results = self.comparison_engine.verify_fields(ocr_results)
            
            log_rx_summary(self.last_rx_number or "", results)
            self.last_verified_signature = self._get_prescription_signature(ocr_results)
            
            matches = sum(1 for r in results.values() if r["match"])
            if matches == len(results) and matches > 0:
                self._handle_all_fields_matched()
            
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

    def run(self):
        """Main monitoring loop."""
        logging.info("Starting to monitor for 'pre-check rx' text...")
        consecutive_no_change = 0
        
        while not self.should_stop:
            try:
                screenshot = pyautogui.screenshot()
                screen_changed = self._has_screen_changed(screenshot)
                if screen_changed:
                    consecutive_no_change = 0
                    logging.debug("Screen activity detected")
                    if self.overlay_root and (time.time() - self.overlay_created_time) > self.advanced_settings.get("overlay", {}).get("min_display_seconds", 3.0):
                        self._close_overlay()
                else:
                    consecutive_no_change += 1
                
                trigger_detected, current_rx_number = self._check_trigger(screenshot)
                now = time.time()
                if trigger_detected:
                    self.last_seen_trigger_time = now
                
                if trigger_detected:
                    should_start = False
                    if not self.recently_triggered:
                        should_start = screen_changed
                    elif current_rx_number and current_rx_number != self.last_rx_number:
                        should_start = True

                    if should_start and current_rx_number:
                        last_time = self.processed_rx_times.get(current_rx_number)
                        cooldown = float(self.config.get("timing", {}).get("same_prescription_wait_seconds", 3.0))
                        if last_time and (now - last_time) < cooldown:
                            logging.info(f"Skipping duplicate Rx within cooldown: #{current_rx_number}")
                            should_start = False

                    if should_start:
                        logging.info(f"New Rx detected: #{current_rx_number}, starting verification...")
                        self.last_rx_number = current_rx_number
                        self.recently_triggered = True
                        self.last_trigger_time = now
                        if current_rx_number:
                            self.processed_rx_times[current_rx_number] = now

                        time.sleep(self.config["timing"]["verification_wait_seconds"])
                        fresh_screenshot = pyautogui.screenshot()
                        
                        ocr_results = self._perform_ocr_on_all_fields(fresh_screenshot)
                        current_sig = self._get_prescription_signature(ocr_results)
                        
                        if self.last_verified_signature and current_sig and current_sig == self.last_verified_signature and (not current_rx_number or current_rx_number == self.last_rx_number):
                            logging.info("Prescription signature unchanged; skipping duplicate verification")
                        else:
                            self._verify_all_fields(fresh_screenshot)
                elif self.recently_triggered:
                    reset_delay = self.advanced_settings.get("trigger", {}).get("lost_reset_delay_seconds", 5.0)
                    if self.last_seen_trigger_time and (now - self.last_seen_trigger_time) > reset_delay:
                        logging.info("Trigger text absent, resetting for next prescription")
                        self.recently_triggered = False
                        self.last_rx_number = None
                        self._close_overlay()

                sleep_time = self.config["timing"]["fast_polling_seconds"] if consecutive_no_change < 10 else self.config["timing"]["max_static_sleep_seconds"]
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                self.stop()
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(1)

def load_config(path: str) -> Optional[Dict[str, Any]]:
    """Loads configuration from a JSON file."""
    try:
        with open(path, 'r') as f:
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
    if config:
        controller = VerificationController(config)
        controller.run()
    else:
        logging.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
