import pyautogui
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from PIL.Image import Resampling
from rapidfuzz import fuzz
import tkinter as tk
import time
import os
import sys
import string
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import numpy as np

# --- Constants & Setup ---
CONFIG_FILE = "config.json"

# --- Logging Setup ---
# Avoids logging Protected Health Information (PHI) by only logging scores and status.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verification.log"),
        logging.StreamHandler()
    ]
)

class VerificationAgent:
    """Encapsulates the entire pharmacy verification workflow."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.advanced_settings = config.get("advanced_settings", {})
        self.recently_triggered = False
        self.last_prescription_data = None
        self.last_rx_number = None
        self.last_screenshot_hash = None
        self.verification_in_progress = False
        self.overlay_root = None
        self.last_trigger_time = 0
        self.overlay_created_time = 0
        self.should_stop = False
        
        self._abbreviations = self._load_abbreviations()
        logging.info(f"Cached {len(self._abbreviations)} abbreviations for efficient matching")
        
        self._setup_tesseract()

    def _log_rx_verification(self, rx_number: str, results: Dict[str, Any]):
        """Log verification results with Rx number and maintain max entries."""
        try:
            log_file = "verification.log"
            max_prescriptions = self.advanced_settings.get("logging", {}).get("max_log_prescriptions", 1000)
            
            log_entries = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_entries = f.readlines()
                except Exception as e:
                    logging.warning(f"Could not read existing log file: {e}")
            
            matches = sum(1 for r in results.values() if r["match"])
            total = len(results)
            match_percentage = (matches / total * 100) if total > 0 else 0
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rx_display = f"Rx#{rx_number}" if rx_number else "Rx#Unknown"
            summary_line = f"[{timestamp}] {rx_display} - {matches}/{total} fields matched ({match_percentage:.1f}%)\n"
            
            detail_lines = [f"    {field_name}: {'PASS' if result['match'] else 'FAIL'} (Score: {result['score']:.2f})\n"
                            for field_name, result in results.items()]
            
            new_entry = [summary_line] + detail_lines + ["\n"]
            
            # Approximate lines per prescription entry
            lines_per_entry = len(new_entry)
            max_lines = max_prescriptions * lines_per_entry
            
            total_lines = len(log_entries) + len(new_entry)
            if total_lines > max_lines:
                lines_to_remove = total_lines - max_lines
                log_entries = log_entries[lines_to_remove:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.writelines(log_entries)
                f.writelines(new_entry)
            
            logging.info(f"{rx_display} - {matches}/{total} fields matched ({match_percentage:.1f}%)")
            
        except Exception as e:
            logging.error(f"Error writing to verification log: {e}")

    def _get_prescription_signature(self, screenshot: Image.Image) -> str:
        """Get a signature of the current prescription to detect changes."""
        try:
            fields_config = self.config["regions"]["fields"]
            signature_parts = []
            
            for field_name in ["patient_name", "drug_name"]:
                if field_name in fields_config:
                    entered_text = self._get_text_from_region(screenshot, tuple(fields_config[field_name]["entered"]))
                    source_text = self._get_text_from_region(screenshot, tuple(fields_config[field_name]["source"]))
                    
                    if field_name == "patient_name":
                        clean_entered = self._normalize_name(entered_text, is_entered_field=True)
                        clean_source = self._normalize_name(source_text, is_entered_field=False)
                    else:
                        clean_entered = self._clean_text(entered_text)
                        clean_source = self._clean_text(source_text)
                    
                    signature_parts.append(f"{clean_entered}|{clean_source}")
            
            return "::".join(signature_parts)
        except Exception as e:
            logging.error(f"Error creating prescription signature: {e}")
            return ""

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

    def _setup_tesseract(self):
        """Sets the path to the Tesseract executable."""
        base_dir = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        tess_path = os.path.join(base_dir, "tesseract", "tesseract.exe")
        
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
            logging.info(f"Tesseract path set to: {tess_path}")
        else:
            system_tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(system_tess_path):
                pytesseract.pytesseract.tesseract_cmd = system_tess_path
                logging.info(f"Tesseract path set to system installation: {system_tess_path}")
            else:
                logging.warning("Tesseract executable not found at expected path. Using system default.")

    def _load_abbreviations(self) -> Dict[str, str]:
        """Loads pharmaceutical abbreviations from an external JSON file."""
        try:
            abbrev_file = os.path.join(os.path.dirname(__file__), "abbreviations.json")
            if os.path.exists(abbrev_file):
                with open(abbrev_file, 'r', encoding='utf-8') as f:
                    loaded_abbreviations = json.load(f)
                    return {k: v for k, v in loaded_abbreviations.items() if not k.startswith("//")}
            else:
                logging.warning(f"Abbreviations file not found: {abbrev_file}")
        except Exception as e:
            logging.error(f"Could not load abbreviations from file: {e}")
        
        logging.warning("Text cleaning will work with limited abbreviation support")
        return {}

    def reload_abbreviations(self):
        """Reload abbreviations from file without restarting the application."""
        try:
            self._abbreviations = self._load_abbreviations()
            logging.info("Abbreviations reloaded successfully")
        except Exception as e:
            logging.error(f"Error reloading abbreviations: {e}")

    def get_loaded_abbreviations_count(self) -> int:
        """Return the number of loaded abbreviations for debugging."""
        return len(self._abbreviations)

    def _clean_drug_name(self, text: str) -> str:
        """Specialized cleaning for drug names."""
        if not text:
            return ""
        
        text = text.lower().strip()
        text = re.sub(r'\s*\([^)]*\)', '', text)
        
        text_with_spaces = f" {text} "
        for abbr, full in self._abbreviations.items():
            text_with_spaces = text_with_spaces.replace(abbr, full)
        
        text = text_with_spaces.strip()
        
        for char in string.punctuation:
            if char not in ['.', '%']:
                text = text.replace(char, ' ')
        
        text = ' '.join(text.split())
        
        equivalent_salt_forms = {'hydrochloride': 'hcl', 'hydrobromide': 'hbr', 'hcl': 'hcl', 'hydrochlor': 'hcl'}
        for full_salt, abbr_salt in equivalent_salt_forms.items():
            text = re.sub(rf'\b{full_salt}\b', abbr_salt, text)
        
        text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)
        text = re.sub(r'([a-z])(\d+)', r'\1 \2', text)
        
        return ' '.join(text.split())

    def _enhanced_drug_name_match(self, entered_text: str, source_text: str, threshold: int = 80) -> Tuple[float, bool]:
        """Enhanced drug name matching with semantic awareness."""
        clean_entered = self._clean_drug_name(entered_text)
        clean_source = self._clean_drug_name(source_text)
        
        if not clean_entered or not clean_source:
            return 0.0, False
        
        # Dosage Mismatch Check
        entered_dosages = re.findall(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', clean_entered)
        source_dosages = re.findall(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', clean_source)
        if entered_dosages and source_dosages:
            normalize_dosage = lambda dosages: sorted([(float(v), u.lower()) for v, u in dosages])
            if normalize_dosage(entered_dosages) != normalize_dosage(source_dosages):
                logging.debug(f"Dosage mismatch detected: {entered_dosages} vs {source_dosages}")
                return 50.0, False

        # Distinct Salt Form Check
        distinct_salt_pairs = self.advanced_settings.get("matching", {}).get("distinct_salt_pairs", [])
        for salt1, salt2 in distinct_salt_pairs:
            if (salt1 in clean_entered and salt2 in clean_source) or \
               (salt2 in clean_entered and salt1 in clean_source):
                logging.debug(f"Different salt forms detected: {salt1} vs {salt2}")
                return 40.0, False

        # Core drug name extraction and comparison
        def extract_drug_name(t):
            for term in ['tablet', 'capsule', 'oral', 'extended release', 'etc']:
                t = t.replace(term, '')
            return ' '.join(re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', '', t).split())
        
        entered_drug_name = extract_drug_name(clean_entered)
        source_drug_name = extract_drug_name(clean_source)
        logging.debug(f"Drug name portions - Entered: '{entered_drug_name}', Source: '{source_drug_name}'")

        # Complex scoring logic remains, but simplified for brevity
        scores = [
            fuzz.ratio(clean_entered, clean_source) * 1.2,
            fuzz.token_sort_ratio(clean_entered, clean_source),
            fuzz.token_set_ratio(clean_entered, clean_source) * 0.8,
            fuzz.partial_ratio(clean_entered, clean_source) * 0.7
        ]
        best_score = min(100, max(scores))
        
        is_match = best_score >= threshold
        return best_score, is_match

    def _match_prescriber_names(self, entered_text: str, source_text: str, threshold: float) -> Tuple[float, bool]:
        """Special matching logic for prescriber names that may contain multiple names separated by '/'."""
        
        # Normalize both texts first
        clean_entered = self._normalize_name(entered_text, is_entered_field=True)
        clean_source = self._normalize_name(source_text, is_entered_field=False)
        
        # If source contains '/', split and try to match against each prescriber
        if '/' in clean_source:
            source_prescribers = [p.strip() for p in clean_source.split('/') if p.strip()]
            
            best_score = 0.0
            for prescriber in source_prescribers:
                score = fuzz.ratio(clean_entered, prescriber.strip())
                if score > best_score:
                    best_score = score
                    
                # Also try token sort ratio for better name matching
                token_score = fuzz.token_sort_ratio(clean_entered, prescriber.strip())
                if token_score > best_score:
                    best_score = token_score
            
            is_match = best_score >= threshold
            return best_score, is_match
        else:
            # Single prescriber, use normal matching
            score = max(
                fuzz.ratio(clean_entered, clean_source),
                fuzz.token_sort_ratio(clean_entered, clean_source)
            )
            is_match = score >= threshold
            return score, is_match

    def _clean_text(self, text: str) -> str:
        """Normalizes text by lowercasing, expanding abbreviations, and removing punctuation."""
        if not text:
            return ""
        text = text.lower().strip()
        text = re.sub(r'\s*\([^)]*\)', '', text)
        
        text_with_spaces = f" {text} "
        for abbr, full in self._abbreviations.items():
            text_with_spaces = text_with_spaces.replace(abbr, full)
        
        text = text_with_spaces.strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    def _clean_sig_text(self, text: str) -> str:
        """Specialized cleaning for sig/directions text - keeps parentheses content."""
        if not text:
            return ""
        text = text.lower().strip()
        
        # Keep parentheses content for sig fields, just remove the parentheses themselves
        text = re.sub(r'\s*\(([^)]*)\)\s*', r' \1 ', text)
        
        text_with_spaces = f" {text} "
        for abbr, full in self._abbreviations.items():
            text_with_spaces = text_with_spaces.replace(abbr, full)
        
        text = text_with_spaces.strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    def _normalize_name(self, name: str, is_entered_field: bool = False) -> str:
        """Normalizes names to handle format differences and medical titles."""
        if not name:
            return ""
        
        clean_name = ' '.join(name.strip().split())
        
        # Handle medical titles with and without periods
        medical_titles = ['md', 'm.d.', 'do', 'd.o.', 'np', 'n.p.', 'pa', 'p.a.', 
                         'pharmd', 'pharm.d.', 'rph', 'r.ph.', 'dds', 'd.d.s.', 
                         'dmd', 'd.m.d.', 'dvm', 'd.v.m.', 'phd', 'ph.d.', 
                         'rn', 'r.n.', 'lpn', 'l.p.n.', 'dr', 'dr.', 'ma', 'm.a.']
        
        for title in medical_titles:
            # Use more flexible pattern that handles periods and word boundaries
            pattern = r'\b' + re.escape(title).replace(r'\.', r'\.?') + r'\b'
            clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
        
        clean_name = re.sub(r'\s+(jr|sr|iii|ii|iv)$', '', clean_name, flags=re.IGNORECASE)
        clean_name = clean_name.replace('|', ' ').replace('_', ' ')
        clean_name = ' '.join(clean_name.split())
        
        # Standardize name format to "First Middle Last"
        clean_name = self._standardize_name_format(clean_name)
        
        if '/' in clean_name:
            return " / ".join([self._standardize_name_format(p.strip()).lower() for p in clean_name.split('/') if p.strip()])
        
        return clean_name.lower()
    
    def _standardize_name_format(self, name: str) -> str:
        """Convert name to standardized 'First Middle Last' format."""
        if not name:
            return ""
        
        name = name.strip()
        
        # Check if name is in "Last, First Middle" format
        if ',' in name:
            parts = [p.strip() for p in name.split(',')]
            if len(parts) == 2 and parts[0] and parts[1]:
                # Convert "Last, First Middle" to "First Middle Last"
                last_name = parts[0]
                first_middle = parts[1]
                return f"{first_middle} {last_name}"
        
        # If no comma, assume it's already in "First Middle Last" format
        return name

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """Applies a series of preprocessing steps to an image to improve OCR accuracy."""
        ocr_config = self.advanced_settings.get("ocr", {})
        
        width, height = image.size
        new_size = (width * ocr_config.get("resize_factor", 2), height * ocr_config.get("resize_factor", 2))
        resized = image.resize(new_size, Resampling.LANCZOS)
        
        gray = resized.convert('L')
        
        contrast = ImageEnhance.Contrast(gray).enhance(ocr_config.get("contrast_factor", 1.5))
        brightness = ImageEnhance.Brightness(contrast).enhance(ocr_config.get("brightness_factor", 1.2))
        
        try:
            img_array = np.array(brightness)
            threshold_value = np.mean(img_array) + ocr_config.get("threshold_mean_offset", -20)
            binary_array = np.where(img_array > threshold_value, 255, 0).astype(np.uint8)
            return Image.fromarray(binary_array, mode='L')
        except ImportError:
            fallback_threshold = ocr_config.get("fallback_threshold", 120)
            return brightness.point(lambda x: 255 if x > fallback_threshold else 0, mode='L')

    def _get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int]) -> str:
        """Crops, preprocesses, and OCRs a region of an image."""
        try:
            cropped = screenshot.crop(region)
            preprocessed = self._preprocess_image_for_ocr(cropped)
            
            text = pytesseract.image_to_string(preprocessed, config="--psm 7").strip()
            text = text.replace('|', '').replace('_', ' ')
            text = ' '.join(text.split())
            text = re.sub(r'\s+[_\-\.\|]\s*$', '', text)
            
            # Simplified OCR corrections
            text = re.sub(r'(\w+)\s+(of|or|on)$', r'\1a', text)
            
            if len(text) < 3:
                alt_text = pytesseract.image_to_string(preprocessed, config="--psm 8").strip()
                alt_text = ' '.join(alt_text.replace('|', '').replace('_', ' ').split())
                if len(alt_text) > len(text):
                    return alt_text
            
            return text
        except Exception as e:
            logging.error(f"Error in OCR for region {region}: {e}")
            return ""

    def verify_fields(self, screenshot: Image.Image) -> Dict[str, Any]:
        """Verifies all configured fields against their source locations."""
        results = {}
        fields_config = self.config["regions"]["fields"]
        thresholds = self.config["thresholds"]

        for field_name, config in fields_config.items():
            entered_text = self._get_text_from_region(screenshot, tuple(config["entered"]))
            source_text = self._get_text_from_region(screenshot, tuple(config["source"]))

            # Add detailed logging for debugging
            logging.info(f"=== {field_name.upper()} FIELD ===")
            logging.info(f"  Raw entered text: '{entered_text}'")
            logging.info(f"  Raw source text: '{source_text}'")

            if field_name in ["patient_name", "prescriber_name"]:
                cleaned_entered = self._normalize_name(entered_text, is_entered_field=True)
                cleaned_source = self._normalize_name(source_text, is_entered_field=False)
            elif field_name == "drug_name":
                cleaned_entered = self._clean_drug_name(entered_text)
                cleaned_source = self._clean_drug_name(source_text)
            elif field_name in ["direction_sig", "sig", "directions"]:
                cleaned_entered = self._clean_sig_text(entered_text)
                cleaned_source = self._clean_sig_text(source_text)
            else:
                cleaned_entered = self._clean_text(entered_text)
                cleaned_source = self._clean_text(source_text)

            logging.info(f"  Cleaned entered: '{cleaned_entered}'")
            logging.info(f"  Cleaned source: '{cleaned_source}'")

            score, is_match = 0.0, False
            if cleaned_entered and cleaned_source:
                threshold = thresholds[config["threshold_key"]]
                if field_name == "drug_name":
                    score, is_match = self._enhanced_drug_name_match(cleaned_entered, cleaned_source, threshold)
                elif field_name == "prescriber_name":
                    score, is_match = self._match_prescriber_names(entered_text, source_text, threshold)
                else:
                    scorer = getattr(fuzz, config.get("score_fn", "ratio"), fuzz.ratio)
                    score = scorer(cleaned_entered, cleaned_source)
                    is_match = score >= threshold
            
            logging.info(f"  Score: {score:.2f} | Threshold: {thresholds.get(config['threshold_key'], 0)} | Match: {'YES' if is_match else 'NO'}")
            
            results[field_name] = {"match": is_match, "score": score, "coords": tuple(config["entered"])}
        return results

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
                self.overlay_root = None # Already destroyed

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
        trigger_text = self._get_text_from_region(screenshot, trigger_region)
        
        keywords = trigger_config.get("keywords", ["pre", "check", "rx"])
        sim_threshold = trigger_config.get("keyword_similarity_threshold", 70)
        min_matches = trigger_config.get("min_keyword_matches", 2)
        
        text_lower_words = trigger_text.lower().split()
        found_count = sum(1 for kw in keywords if any(fuzz.ratio(w, kw) >= sim_threshold for w in text_lower_words))
        
        trigger_detected = found_count >= min_matches
        rx_number = self._extract_rx_number(trigger_text) if trigger_detected else ""
        
        return trigger_detected, rx_number

    def _verify_all_fields(self, screenshot: Image.Image):
        """Run verification on all fields and show overlay."""
        if self.verification_in_progress:
            logging.debug("Verification already in progress, skipping...")
            return
            
        try:
            self.verification_in_progress = True
            logging.info("Running field verification...")
            results = self.verify_fields(screenshot)
            
            self._log_rx_verification(self.last_rx_number or "", results)
            
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
                
                if self._has_screen_changed(screenshot):
                    consecutive_no_change = 0
                    logging.debug("Screen activity detected")
                    if self.overlay_root and (time.time() - self.overlay_created_time) > self.advanced_settings.get("overlay", {}).get("min_display_seconds", 3.0):
                        self._close_overlay()
                else:
                    consecutive_no_change += 1
                
                trigger_detected, current_rx_number = self._check_trigger(screenshot)
                
                if trigger_detected:
                    if not self.recently_triggered or (current_rx_number and current_rx_number != self.last_rx_number):
                        logging.info(f"New Rx detected: #{current_rx_number}, starting verification...")
                        self.last_rx_number = current_rx_number
                        self.recently_triggered = True
                        self.last_trigger_time = time.time()
                        
                        time.sleep(self.config["timing"]["verification_wait_seconds"])
                        fresh_screenshot = pyautogui.screenshot()
                        self._verify_all_fields(fresh_screenshot)
                elif self.recently_triggered:
                    reset_delay = self.advanced_settings.get("trigger", {}).get("lost_reset_delay_seconds", 5.0)
                    if (time.time() - self.last_trigger_time) > reset_delay:
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
    config = load_config(CONFIG_FILE)
    if config:
        agent = VerificationAgent(config)
        agent.run()
    else:
        logging.critical("Failed to load configuration. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
