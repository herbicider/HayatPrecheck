
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

from ocr_provider import TesseractOcrProvider, EasyOcrProvider, get_cached_ocr_provider
from comparison_engine import ComparisonEngine
from logger_config import setup_logging, log_rx_summary

class VerificationController:
    """Manages the main application loop, screen monitoring, and UI."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.advanced_settings = config.get("advanced_settings", {})
        
        # Initialize OCR provider based on config (using cache to avoid reinitialization)
        ocr_provider_type = config.get("ocr_provider", "tesseract")
        self.ocr_provider = get_cached_ocr_provider(ocr_provider_type, self.advanced_settings)
        logging.info(f"Using OCR provider: {ocr_provider_type}")
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
            
            # Use mandatory fields for the core signature
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

    def _extract_rx_number(self, screenshot: Image.Image) -> str:
        """Extract the Rx number from the rx_number region."""
        try:
            # Use separate rx_number region if available, otherwise fall back to trigger region
            rx_region = self.config["regions"].get("rx_number")
            if rx_region is None:
                rx_region = self.config["regions"]["trigger"]
            
            rx_text = self.ocr_provider.get_text_from_region(screenshot, tuple(rx_region))
            
            patterns = [r'rx\s*-\s*(\d+)', r'rx\s+(\d+)', r'(\d{4,})']
            text_lower = rx_text.lower()
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(1)
            return ""
        except Exception as e:
            logging.error(f"Error extracting Rx number: {e}")
            return ""

    def _check_trigger(self, screenshot: Image.Image) -> Tuple[bool, str]:
        """Check if the trigger text is present with enhanced keyword matching and OCR fallback."""
        trigger_config = self.advanced_settings.get("trigger", {})
        trigger_region = tuple(self.config["regions"]["trigger"])
        trigger_text = self.ocr_provider.get_text_from_region(screenshot, trigger_region)
        
        keywords = trigger_config.get("keywords", ["pre", "check", "rx"])
        sim_threshold = trigger_config.get("keyword_similarity_threshold", 90)
        min_matches = trigger_config.get("min_keyword_matches", 2)
        
        # Debug: Log trigger check details (reduced frequency)
        # Only log every 20 iterations to reduce spam
        if hasattr(self, '_trigger_check_count'):
            self._trigger_check_count += 1
        else:
            self._trigger_check_count = 1
            
        if self._trigger_check_count % 20 == 0:
            logging.info(f"🔍 Checking trigger: '{trigger_text}' | Keywords: {keywords} | Need {min_matches} matches | Threshold: {sim_threshold}%")
        
        from rapidfuzz import fuzz
        
        # Process text for matching
        text_lower = trigger_text.lower()
        
        # Check if keywords should be treated as a single phrase
        if len(keywords) > 1:
            # Try matching the complete phrase first
            full_phrase = " ".join(keywords)
            phrase_similarity = fuzz.ratio(text_lower.strip(), full_phrase.lower())
            
            # Temporarily lower threshold to test
            if phrase_similarity >= 80:  # Lowered from sim_threshold to 80 for testing
                logging.info(f"✅ Full phrase match found: {phrase_similarity}%")
                trigger_detected = True
                rx_number = self._extract_rx_number(screenshot)
                logging.info(f"✅ Trigger detected: '{trigger_text}' | Full phrase match: {phrase_similarity}% | Rx#: '{rx_number or 'None'}'")
                return trigger_detected, rx_number
            else:
                logging.info(f"❌ Phrase match failed: {phrase_similarity}% < 80%")
        else:
            logging.info(f"🔧 Skipping phrase matching - only {len(keywords)} keyword(s)")
        
        # Fall back to individual keyword matching if phrase matching fails
        
        # Split text by both spaces and common separators to handle "pre-check" cases
        import re
        text_words = re.split(r'[\s\-_.,;:|"\']+', text_lower)
        text_words = [w for w in text_words if w]  # Remove empty strings
        
        found_count = 0
        matched_keywords = []
        
        for kw in keywords:
            best_match = 0
            best_word = ""
            
            # Check direct word matches
            for w in text_words:
                score = fuzz.ratio(w, kw.lower())  # Make keyword lowercase for comparison
                if score > best_match:
                    best_match = score
                    best_word = w
            
            # Also check substring matches for compound words like "pre-check"
            if best_match < sim_threshold:
                kw_lower = kw.lower()  # Convert keyword to lowercase once
                for w in text_words:
                    if kw_lower in w or w in kw_lower:
                        substring_score = max(
                            fuzz.ratio(kw_lower, w),
                            fuzz.partial_ratio(kw_lower, w),
                            fuzz.partial_ratio(w, kw_lower)
                        )
                        if substring_score > best_match:
                            best_match = substring_score
                            best_word = w
            
            # Check against the full text for phrases like "pre-check"
            if best_match < sim_threshold:
                full_text_score = fuzz.partial_ratio(kw.lower(), text_lower)
                if full_text_score > best_match:
                    best_match = full_text_score
                    best_word = f"(in full text)"
            
            if best_match >= sim_threshold:
                found_count += 1
                matched_keywords.append(f"{kw}→{best_word}({best_match:.1f})")
        
        trigger_detected = found_count >= min_matches
        rx_number = self._extract_rx_number(screenshot) if trigger_detected else ""
        
        # Debug logging for trigger detection
        if trigger_detected:
            logging.info(f"✅ Trigger detected: '{trigger_text}' | Matched: {matched_keywords} | Rx#: '{rx_number or 'None'}'")
        else:
            # Only log failures occasionally to avoid spam
            if self._trigger_check_count % 50 == 0:
                logging.info(f"❌ Trigger not detected: '{trigger_text}' | Found {found_count}/{min_matches} keywords: {matched_keywords}")
        
        return trigger_detected, rx_number
    
    def _is_suspicious_ocr_result(self, text: str) -> bool:
        """Check if OCR result looks suspicious (garbage text)."""
        if not text or len(text.strip()) < 3:
            return True
            
        # Check for patterns that indicate garbage OCR
        # 1. Too many single characters
        words = text.split()
        single_chars = sum(1 for w in words if len(w) == 1)
        if len(words) > 3 and single_chars / len(words) > 0.6:
            return True
            
        # 2. Repetitive characters like "s s s s s" or "h x s s s s e h s s s s s s"
        if len(set(text.replace(' ', '').lower())) < 3 and len(text) > 10:
            return True
            
        # 3. Repetitive characters like "s s s s s" pattern
        if len(set(text.replace(' ', '').lower())) < 3 and len(text) > 10:
            return True
            
        # 4. No recognizable English patterns
        import re
        if not re.search(r'[a-zA-Z]{2,}', text):  # No words with 2+ letters
            return True
            
        return False
    
    def _ocr_with_retry(self, screenshot: Image.Image, region: Tuple[int, int, int, int], field_identifier: str, max_retries: Optional[int] = None) -> str:
        """
        Perform OCR with retry mechanism for empty results.
        
        Args:
            screenshot: PIL Image to process
            region: Tuple of (x1, y1, x2, y2) coordinates
            field_identifier: String identifier for logging
            max_retries: Maximum number of retry attempts (uses config if None)
            
        Returns:
            OCR text result, empty string if all retries fail
        """
        retry_config = self.config.get("timing", {})
        retry_delay = retry_config.get("ocr_retry_delay_seconds", 0.5)
        
        # Use configured retry count if not specified
        if max_retries is None:
            max_retries = int(retry_config.get("ocr_max_retries", 3))
        else:
            max_retries = int(max_retries)
        
        for attempt in range(max_retries):
            try:
                # Perform OCR
                text = self.ocr_provider.get_text_from_region(screenshot, region, field_identifier)
                
                # Check if we got meaningful text
                if text and text.strip():
                    if attempt > 0:
                        logging.info(f"OCR retry success for {field_identifier} on attempt {attempt + 1}: '{text[:50]}...'")
                    return text
                else:
                    if attempt < max_retries - 1:  # Not the last attempt
                        logging.warning(f"OCR attempt {attempt + 1} for {field_identifier} returned empty. Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"OCR failed for {field_identifier} after {max_retries} attempts. Final result: empty")
                        
            except Exception as e:
                logging.error(f"OCR attempt {attempt + 1} for {field_identifier} failed with error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
        
        return ""  # All attempts failed
    
    def _perform_ocr_on_all_fields(self, screenshot: Image.Image) -> Dict[str, Tuple[str, str]]:
        """Performs OCR on all configured fields and returns the raw text."""
        logging.info("Starting OCR processing on all fields...")
        ocr_results = {}
        
        # Include mandatory fields plus any enabled optional fields
        fields_to_process = list(self.config["regions"]["fields"].keys())
        enabled_optional_fields = self.config.get("optional_fields_enabled", {})
        
        for field_name in list(fields_to_process):
            if field_name not in ["patient_name", "prescriber_name", "drug_name", "direction_sig"]:
                if not enabled_optional_fields.get(field_name, False):
                    fields_to_process.remove(field_name)

        for field_name in fields_to_process:
            config = self.config["regions"]["fields"][field_name]
            logging.info(f"Processing OCR for field: {field_name}")
            try:
                logging.debug(f"OCR for {field_name} entered region: {config['entered']}")
                entered_text = self._ocr_with_retry(screenshot, tuple(config["entered"]), f"{field_name}_entered")
                
                logging.debug(f"OCR for {field_name} source region: {config['source']}")
                source_text = self._ocr_with_retry(screenshot, tuple(config["source"]), f"{field_name}_source")
                
                ocr_results[field_name] = (entered_text, source_text)
                logging.info(f"Completed OCR for field: {field_name} | Entered: '{entered_text[:50]}...' | Source: '{source_text[:50]}...'")
                
                # Log warning if either field is empty after retries
                if not entered_text.strip():
                    logging.warning(f"Field {field_name} entered text is empty after OCR retries")
                if not source_text.strip():
                    logging.warning(f"Field {field_name} source text is empty after OCR retries")
                    
            except Exception as e:
                logging.error(f"Error processing OCR for {field_name}: {e}")
                ocr_results[field_name] = ("", "")
        
        logging.info(f"Completed OCR processing for {len(ocr_results)} fields")
        return ocr_results

    def _verify_all_fields(self, screenshot: Image.Image, ocr_results: Optional[Dict[str, Tuple[str, str]]] = None):
        """Run verification on all fields and show overlay."""
        if self.verification_in_progress:
            logging.debug("Verification already in progress, skipping...")
            return
            
        try:
            self.verification_in_progress = True
            logging.info("Running field verification...")
            
            # Use provided OCR results or perform OCR if not provided
            if ocr_results is None:
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
        loop_count = 0
        
        while not self.should_stop:
            try:
                loop_count += 1
                screenshot = pyautogui.screenshot()
                
                if loop_count % 10 == 0:  # Log every 10th iteration to show it's running
                    # Add debug info showing what's in the trigger area
                    trigger_region = tuple(self.config["regions"]["trigger"])
                    trigger_text = self.ocr_provider.get_text_from_region(screenshot, trigger_region)
                    logging.info(f"Monitoring loop running... (iteration {loop_count}) | Trigger area reads: '{trigger_text.strip()}'")
                
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
                        # Start verification on trigger detection, regardless of screen change
                        should_start = True
                        logging.info(f"🎯 First trigger detection - starting verification")
                    elif current_rx_number and current_rx_number != self.last_rx_number:
                        should_start = True
                        logging.info(f"🔄 New Rx number detected: {current_rx_number}")
                    elif not current_rx_number and not self.last_rx_number:
                        # Allow trigger without Rx number if we haven't processed anything recently
                        should_start = True
                        logging.info(f"🎯 Trigger detected without Rx number - starting verification")

                    # Check cooldown only if we have an Rx number
                    if should_start and current_rx_number:
                        last_time = self.processed_rx_times.get(current_rx_number)
                        cooldown = float(self.config.get("timing", {}).get("same_prescription_wait_seconds", 3.0))
                        if last_time and (now - last_time) < cooldown:
                            logging.info(f"Skipping duplicate Rx within cooldown: #{current_rx_number}")
                            should_start = False

                    if should_start:
                        rx_display = f"#{current_rx_number}" if current_rx_number else "without Rx number"
                        logging.info(f"New prescription detected: {rx_display}, starting verification...")
                        self.last_rx_number = current_rx_number
                        self.recently_triggered = True
                        self.last_trigger_time = now
                        if current_rx_number:
                            self.processed_rx_times[current_rx_number] = now

                        # Add configurable delay to allow content to fully load after trigger
                        content_load_delay = self.config.get("timing", {}).get("trigger_content_load_delay_seconds", 0.5)
                        logging.debug(f"Waiting {content_load_delay}s for content to load after trigger detection...")
                        time.sleep(content_load_delay)
                        
                        time.sleep(self.config["timing"]["verification_wait_seconds"])
                        logging.info("Taking fresh screenshot for verification...")
                        fresh_screenshot = pyautogui.screenshot()
                        
                        logging.info("Starting OCR on all fields...")
                        ocr_results = self._perform_ocr_on_all_fields(fresh_screenshot)
                        logging.info("Getting prescription signature...")
                        current_sig = self._get_prescription_signature(ocr_results)
                        
                        if self.last_verified_signature and current_sig and current_sig == self.last_verified_signature and (not current_rx_number or current_rx_number == self.last_rx_number):
                            logging.info("Prescription signature unchanged; skipping duplicate verification")
                        else:
                            self._verify_all_fields(fresh_screenshot, ocr_results)
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
    try:
        setup_logging()
        config = load_config("config.json")
        if config:
            controller = VerificationController(config)
            controller.run()
        else:
            logging.critical("Failed to load configuration. Exiting.")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error in main(): {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
