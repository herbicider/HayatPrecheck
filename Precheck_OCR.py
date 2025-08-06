import pyautogui
import pytesseract
from PIL import Image, ImageEnhance
from PIL.Image import Resampling
from rapidfuzz import fuzz
import tkinter as tk
import time
import os
import sys
import string
import json
import logging
from typing import Dict, Any, Tuple, Optional

# --- Constants & Setup ---
CONFIG_FILE = "config.json"

# --- Logging Setup ---
# Avoids logging Protected Health Information (PHI) by only logging scores and status.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("verification.log"),
        logging.StreamHandler() # Also print to console
    ]
)

class VerificationAgent:
    """Encapsulates the entire pharmacy verification workflow."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.recently_triggered = False
        self.last_prescription_data = None  # Store last prescription data to detect changes
        self.last_rx_number = None  # Store last Rx number to detect changes
        self.last_screenshot_hash = None  # Store hash of last screenshot for change detection
        self.verification_in_progress = False  # Flag to prevent overlapping verifications
        self.overlay_root = None  # Store reference to active overlay
        self.last_trigger_time = 0  # Track when trigger was last processed to prevent rapid re-triggering
        self.overlay_created_time = 0  # Track when overlay was created
        self.should_stop = False  # Flag to stop the monitoring loop
        self._setup_tesseract()

    def _log_rx_verification(self, rx_number: str, results: Dict[str, Any]):
        """Log verification results with Rx number and maintain max 1000 entries."""
        try:
            import os
            
            log_file = "verification.log"
            
            # Read existing log entries
            log_entries = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_entries = f.readlines()
                except Exception as e:
                    logging.warning(f"Could not read existing log file: {e}")
            
            # Create new log entry
            matches = sum(1 for r in results.values() if r["match"])
            total = len(results)
            match_percentage = (matches / total * 100) if total > 0 else 0
            
            # Create detailed entry with timestamp and Rx number
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            rx_display = f"Rx#{rx_number}" if rx_number else "Rx#Unknown"
            summary_line = f"[{timestamp}] {rx_display} - {matches}/{total} fields matched ({match_percentage:.1f}%)\n"
            
            # Add field-by-field details
            detail_lines = []
            for field_name, result in results.items():
                status = "✓" if result["match"] else "✗"
                detail_lines.append(f"    {field_name}: {status} (Score: {result['score']:.2f})\n")
            
            # Combine all lines for this prescription
            new_entry = [summary_line] + detail_lines + ["\n"]  # Add blank line separator
            
            # Keep only last 1000 prescriptions (approximately)
            # Each prescription takes about 6 lines (1 summary + 4 fields + 1 blank)
            max_lines = 1000 * 6  # Rough estimate
            
            # If we have too many lines, remove from the beginning
            total_lines = len(log_entries) + len(new_entry)
            if total_lines > max_lines:
                # Remove oldest entries (lines from beginning)
                lines_to_remove = total_lines - max_lines
                log_entries = log_entries[lines_to_remove:]
            
            # Write back to file
            with open(log_file, 'w', encoding='utf-8') as f:
                f.writelines(log_entries)
                f.writelines(new_entry)
            
            # Also log to standard logger for console output
            logging.info(f"{rx_display} - {matches}/{total} fields matched ({match_percentage:.1f}%)")
            
        except Exception as e:
            logging.error(f"Error writing to verification log: {e}")

    def _get_prescription_signature(self, screenshot: Image.Image) -> str:
        """Get a signature of the current prescription to detect changes."""
        try:
            # Get text from key fields to create a signature
            fields_config = self.config["regions"]["fields"]
            signature_parts = []
            
            for field_name in ["patient_name", "drug_name"]:
                if field_name in fields_config:
                    entered_text = self._get_text_from_region(screenshot, tuple(fields_config[field_name]["entered"]))
                    source_text = self._get_text_from_region(screenshot, tuple(fields_config[field_name]["source"]))
                    
                    # Apply appropriate normalization
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
        """Get a quick hash of the screenshot to detect changes (NO OCR, just pixels)."""
        try:
            # Crop to a larger, more stable area that includes key regions
            # Use a bigger area to reduce sensitivity to minor pixel changes
            left = 50
            top = 150
            right = min(screenshot.width, 800)  # Much wider area
            bottom = min(screenshot.height, 500)  # Much taller area
            
            cropped = screenshot.crop((left, top, right, bottom))
            
            # Convert to grayscale and resize to a larger size for more stable hashing
            # Larger hash size reduces false positives from minor changes
            small_image = cropped.convert('L').resize((32, 32))  # Smaller hash = less sensitive
            
            # Apply slight blur to reduce noise sensitivity
            from PIL import ImageFilter
            small_image = small_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Create a simple hash from pixel values (NO OCR HERE!)
            pixels = list(small_image.getdata())
            
            # Group pixels into buckets to reduce sensitivity
            # This makes the hash more stable to minor changes
            bucket_size = 8
            bucketed_pixels = []
            for pixel in pixels:
                bucketed_pixels.append(pixel // bucket_size * bucket_size)
            
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
        """Sets the path to the Tesseract executable, especially for PyInstaller."""
        if hasattr(sys, '_MEIPASS'):
            # Running in a PyInstaller bundle
            base_dir = sys._MEIPASS
        else:
            # Running in a normal Python environment
            base_dir = os.path.dirname(os.path.abspath(__file__))

        tess_path = os.path.join(base_dir, "tesseract", "tesseract.exe")
        if os.path.exists(tess_path):
            pytesseract.pytesseract.tesseract_cmd = tess_path
            logging.info(f"Tesseract path set to: {tess_path}")
        else:
            # Try system installation path
            system_tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            if os.path.exists(system_tess_path):
                pytesseract.pytesseract.tesseract_cmd = system_tess_path
                logging.info(f"Tesseract path set to system installation: {system_tess_path}")
            else:
                logging.warning("Tesseract executable not found at expected path. Using system default.")

    def _clean_text(self, text: str) -> str:
        """Normalizes text by lowercasing, expanding abbreviations, and removing punctuation."""
        if not text:
            return ""
        text = text.lower().strip()
        
        # Remove brand names in parentheses (e.g., "Metformin (Glucophage)" -> "Metformin")
        import re
        text = re.sub(r'\s*\([^)]*\)', '', text)
        
        # More comprehensive list of pharmacy abbreviations
        abbreviations = {
            " po ": " by mouth ", " qd": " daily", " od ": " once daily ", " bid": " twice daily",
            " tid": " three times daily", " qid": " four times daily", " ac ": " before meals ",
            " pc ": " after meals ", " hs ": " at bedtime ", " prn": " as needed", " ud ": " as directed ",
            " tab ": " tablet ", " cap ": " capsule ", " gtt ": " drop ", " sol ": " solution ",
            " supp ": " suppository ", " ung ": " ointment ", " disp ": " dispense ",
            " sig ": " directions ", " rx ": " prescription "
        }
        for abbr, full in abbreviations.items():
            text = text.replace(abbr, full)
        
        # Remove punctuation and consolidate whitespace
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    def _normalize_name(self, name: str, is_entered_field: bool = False) -> str:
        """
        Normalizes names to handle format differences and medical titles.
        Extracts only first and last names, skipping middle names for better matching.
        
        Args:
            name: The name to normalize
            is_entered_field: True if this is from the left side (entered field), 
                            False if from right side (source field)
        
        Returns:
            Normalized name in "FirstName LastName" format without titles or middle names
        """
        if not name:
            return ""
        
        # Basic cleaning (remove extra spaces, but keep commas for parsing)
        clean_name = ' '.join(name.strip().split())
        
        # Remove common medical titles (case insensitive)
        medical_titles = [
            'MD', 'M.D.', 'DO', 'D.O.', 'NP', 'N.P.', 'APNP', 'A.P.N.P.',
            'PA', 'P.A.', 'PharmD', 'Pharm.D.', 'RPh', 'R.Ph.', 'DDS', 'D.D.S.',
            'DMD', 'D.M.D.', 'DVM', 'D.V.M.', 'PhD', 'Ph.D.', 'RN', 'R.N.',
            'LPN', 'L.P.N.', 'CNP', 'C.N.P.', 'FNP', 'F.N.P.', 'ANP', 'A.N.P.',
            'Dr.', 'Dr', 'Doctor', 'DNP', 'D.N.P.', 'MSN', 'M.S.N.', 'BSN', 'B.S.N.',
            'CRNA', 'C.R.N.A.', 'APRN', 'A.P.R.N.'
        ]
        
        # Create a pattern to remove titles (with word boundaries)
        import re
        for title in medical_titles:
            # Remove at beginning (e.g., "Dr. John Smith" -> "John Smith")
            pattern_start = r'\b' + re.escape(title) + r'\s+'
            clean_name = re.sub(pattern_start, '', clean_name, flags=re.IGNORECASE)
            
            # Remove at end (e.g., "John Smith MD" -> "John Smith")
            pattern_end = r'\s+' + re.escape(title) + r'\b'
            clean_name = re.sub(pattern_end, '', clean_name, flags=re.IGNORECASE)
            
            # Remove in middle when followed by comma (e.g., "John Smith, MD, PhD" -> "John Smith, PhD")
            pattern_middle = r'\s*,\s*' + re.escape(title) + r'\s*,\s*'
            clean_name = re.sub(pattern_middle, ', ', clean_name, flags=re.IGNORECASE)
            
            # Remove when it's the only thing after comma (e.g., "John Smith, MD" -> "John Smith")
            pattern_comma_end = r'\s*,\s*' + re.escape(title) + r'\s*$'
            clean_name = re.sub(pattern_comma_end, '', clean_name, flags=re.IGNORECASE)
        
        # Clean up any extra spaces and commas after title removal
        clean_name = re.sub(r'\s*,\s*,\s*', ', ', clean_name)  # Fix double commas
        clean_name = re.sub(r'\s*,\s*$', '', clean_name)  # Remove trailing comma
        clean_name = ' '.join(clean_name.strip().split())
        
        # Extract first and last names only (skip middle names)
        if is_entered_field:
            # Left side format: "LastName, FirstName [MiddleName]" or "LastName. FirstName [of/etc]"
            # Handle both comma and period separators
            parts = []
            separator_found = False
            
            if ',' in clean_name:
                parts = clean_name.split(',', 1)
                separator_found = True
            elif '.' in clean_name:
                parts = clean_name.split('.', 1)
                separator_found = True
            
            if separator_found and len(parts) == 2:
                last_name = parts[0].strip()
                first_part = parts[1].strip()
                
                # Remove common suffixes like "of", "jr", "sr", etc.
                import re
                first_part = re.sub(r'\s+(of|jr|sr|iii|ii|iv)$', '', first_part, flags=re.IGNORECASE)
                
                # Extract only the first word as the first name (skip middle names)
                first_name = first_part.split()[0] if first_part else ""
                if first_name and last_name:
                    return f"{first_name} {last_name}".lower()
            
            # If no separator found, try to extract first and last from space-separated format
            name_parts = clean_name.split()
            # Filter out common suffixes
            name_parts = [part for part in name_parts if part.lower() not in ['of', 'jr', 'sr', 'iii', 'ii', 'iv']]
            
            if len(name_parts) >= 2:
                # Assume first word is first name, last word is last name
                return f"{name_parts[0]} {name_parts[-1]}".lower()
            return clean_name.lower()
        else:
            # Right side format: "FirstName [MiddleName] LastName"
            # Handle multiple prescribers separated by "/"
            if '/' in clean_name:
                # Split by "/" and process each prescriber separately
                prescribers = clean_name.split('/')
                processed_prescribers = []
                
                for prescriber in prescribers:
                    prescriber = prescriber.strip()
                    if prescriber:
                        # Apply the same cleaning logic to each prescriber individually
                        prescriber = prescriber.replace('|', ' ')  # Remove pipe separators
                        prescriber = prescriber.replace('_', ' ')  # Remove underscores (OCR artifacts)
                        prescriber = ' '.join(prescriber.split())  # Clean up extra spaces
                        
                        name_parts = prescriber.split()
                        # Filter out empty parts, single characters, and any remaining titles
                        filtered_parts = []
                        common_titles = ['dr', 'md', 'do', 'np', 'pa', 'dnp', 'phd', 'rn', 'lpn', 'cnp', 'fnp', 'anp', 'aprn', 'crna', 'msn', 'bsn']
                        
                        for part in name_parts:
                            # Skip if it's too short, or if it's a title, or if it ends with a period (likely title)
                            if (len(part) > 1 and 
                                part.lower().rstrip('.') not in common_titles and 
                                (not part.endswith('.') or len(part) > 3)):  # Allow longer words even with periods
                                filtered_parts.append(part)
                        
                        if len(filtered_parts) >= 2:
                            # Take first word as first name, last word as last name (skip middle)
                            first_name = filtered_parts[0]
                            last_name = filtered_parts[-1]
                            processed_prescribers.append(f"{first_name} {last_name}".lower())
                        elif len(filtered_parts) == 1:
                            # Only one name part - use as is
                            processed_prescribers.append(filtered_parts[0].lower())
                
                # Return all prescribers joined by " / " for later processing
                if processed_prescribers:
                    return " / ".join(processed_prescribers)
                else:
                    return clean_name.lower()
            else:
                # Single prescriber - use existing logic
                clean_name = clean_name.replace('|', ' ')  # Remove pipe separators
                clean_name = clean_name.replace('_', ' ')  # Remove underscores (OCR artifacts)
                clean_name = ' '.join(clean_name.split())  # Clean up extra spaces
                
                name_parts = clean_name.split()
                # Filter out empty parts, single characters, and any remaining titles
                filtered_parts = []
                common_titles = ['dr', 'md', 'do', 'np', 'pa', 'dnp', 'phd', 'rn', 'lpn', 'cnp', 'fnp', 'anp', 'aprn', 'crna', 'msn', 'bsn']
                
                for part in name_parts:
                    # Skip if it's too short, or if it's a title, or if it ends with a period (likely title)
                    if (len(part) > 1 and 
                        part.lower().rstrip('.') not in common_titles and 
                        (not part.endswith('.') or len(part) > 3)):  # Allow longer words even with periods
                        filtered_parts.append(part)
                
                if len(filtered_parts) >= 2:
                    # Take first word as first name, last word as last name (skip middle)
                    first_name = filtered_parts[0]
                    last_name = filtered_parts[-1]
                    return f"{first_name} {last_name}".lower()
                elif len(filtered_parts) == 1:
                    # Only one name part - return as is
                    return filtered_parts[0].lower()
                return clean_name.lower()

    def _get_text_from_region(self, screenshot: Image.Image, region: Tuple[int, int, int, int]) -> str:
        """Crops a region, applies pre-processing, and performs OCR."""
        try:
            cropped_image = screenshot.crop(region)
            
            # --- Pre-processing for better OCR results ---
            # 1. Resize the image 2x (improves DPI for OCR engine)
            width, height = cropped_image.size
            resized_image = cropped_image.resize((width * 2, height * 2), Resampling.LANCZOS)
            
            # 2. Convert to grayscale first to eliminate color highlighting issues
            gray_image = resized_image.convert('L')
            
            # 3. Apply contrast enhancement to deal with highlighted text
            # This helps make highlighted text more readable
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(gray_image)
            enhanced_image = enhancer.enhance(1.5)  # Increase contrast by 50%
            
            # 4. Apply brightness adjustment to normalize highlighted areas
            brightness_enhancer = ImageEnhance.Brightness(enhanced_image)
            bright_image = brightness_enhancer.enhance(1.2)  # Increase brightness slightly
            
            # 5. Apply threshold to create better black/white separation
            # This helps OCR distinguish text from background better
            try:
                import numpy as np
                img_array = np.array(bright_image)
                
                # Use a more aggressive threshold for highlighted text
                # Lower threshold to capture text that might be partially highlighted
                threshold_value = np.mean(img_array) - 20  # Below average to catch highlighted text
                binary_array = np.where(img_array > threshold_value, 255, 0).astype(np.uint8)
                
                # Convert back to PIL Image
                final_image = Image.fromarray(binary_array, mode='L')
            except ImportError:
                # Fallback: use PIL's point method for thresholding if numpy not available
                def threshold_func(x):
                    return 255 if x > 120 else 0  # Lower threshold for highlighted text
                final_image = bright_image.point(threshold_func, mode='L')

            # 6. Try multiple OCR configurations for better results
            # First try PSM 7 (single line)
            config_str = "--psm 7"
            text = pytesseract.image_to_string(final_image, config=config_str).strip()
            
            # Clean up common OCR artifacts from highlighting
            text = text.replace('|', '')  # Remove pipe characters
            text = text.replace('_', ' ')  # Remove underscores (OCR artifacts)
            text = ' '.join(text.split())  # Clean up extra spaces
            
            # Remove trailing single characters that are likely OCR artifacts
            import re
            text = re.sub(r'\s+[_\-\.\|]\s*$', '', text)  # Remove trailing single punctuation
            
            # Fix common OCR misreadings for names
            # "name of" -> "namea" (OCR often reads final 'a' as 'of')
            text = re.sub(r'(\w+)\s+of$', r'\1a', text)
            # "name or" -> "namea" (similar misreading)
            text = re.sub(r'(\w+)\s+or$', r'\1a', text)
            # "name on" -> "namea" (another variant)
            text = re.sub(r'(\w+)\s+on$', r'\1a', text)
            
            text = ' '.join(text.split())  # Clean up again
            
            # If text is too short or has obvious OCR errors, try PSM 8 (single word)
            if len(text) < 3:
                config_str = "--psm 8"
                text_alt = pytesseract.image_to_string(final_image, config=config_str).strip()
                text_alt = text_alt.replace('|', '')  # Clean this too
                text_alt = text_alt.replace('_', ' ')  # Clean underscores
                text_alt = re.sub(r'\s+[_\-\.\|]\s*$', '', text_alt)  # Remove trailing artifacts
                
                # Apply OCR corrections to alternative text too
                text_alt = re.sub(r'(\w+)\s+of$', r'\1a', text_alt)
                text_alt = re.sub(r'(\w+)\s+or$', r'\1a', text_alt)
                text_alt = re.sub(r'(\w+)\s+on$', r'\1a', text_alt)
                
                text_alt = ' '.join(text_alt.split())
                if len(text_alt) > len(text):  # Use longer result
                    text = text_alt
            
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

            # DEBUG: Show raw OCR text before processing
            print(f"\n=== DEBUG {field_name.upper()} ===")
            print(f"Raw LEFT (entered): '{entered_text}'")
            print(f"Raw RIGHT (source): '{source_text}'")

            # Apply appropriate normalization based on field type
            if field_name in ["patient_name", "prescriber_name"]:
                # Use name normalization for name fields
                cleaned_entered = self._normalize_name(entered_text, is_entered_field=True)
                cleaned_source = self._normalize_name(source_text, is_entered_field=False)
            else:
                # Use regular text cleaning for other fields
                cleaned_entered = self._clean_text(entered_text)
                cleaned_source = self._clean_text(source_text)

            # DEBUG: Show cleaned text after processing
            print(f"Cleaned LEFT: '{cleaned_entered}'")
            print(f"Cleaned RIGHT: '{cleaned_source}'")

            score = 0
            is_match = False
            threshold = 0  # Initialize threshold for debug display
            
            if cleaned_entered and cleaned_source:
                scorer = getattr(fuzz, config["score_fn"], fuzz.ratio)
                threshold_key = config["threshold_key"]
                threshold = thresholds[threshold_key]
                
                # Special handling for prescriber names - check multiple prescribers separated by "/"
                if field_name == "prescriber_name" and " / " in cleaned_source:
                    prescribers = [p.strip() for p in cleaned_source.split(" / ")]
                    best_score = 0
                    for prescriber in prescribers:
                        if prescriber:  # Skip empty strings
                            current_score = scorer(cleaned_entered, prescriber)
                            if current_score > best_score:
                                best_score = current_score
                            # If any prescriber matches the threshold, it's a match
                            if current_score >= threshold:
                                is_match = True
                                break
                    score = best_score
                else:
                    # Regular comparison for other fields or single prescriber
                    score = scorer(cleaned_entered, cleaned_source)
                    is_match = score >= threshold

            # DEBUG: Show final comparison results
            print(f"Score: {score:.2f} | Threshold: {threshold} | Match: {is_match}")
            print("=" * 40)

            results[field_name] = {
                "match": is_match,
                "score": score,
                "coords": tuple(config["entered"])
            }
            # Field-level logging removed to avoid duplication - now handled in _log_rx_verification
        
        return results

    def _show_tk_overlay(self, results: Dict[str, Any]):
        """Displays a transparent overlay with colored rectangles."""
        try:
            # Close existing overlay if present
            if self.overlay_root:
                try:
                    self.overlay_root.destroy()
                except:
                    pass
                self.overlay_root = None
            
            root = tk.Tk()
            self.overlay_root = root  # Store reference for later cleanup
            self.overlay_created_time = time.time()  # Track when overlay was created
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
                coords = result["coords"]
                canvas.create_rectangle(coords[0], coords[1], coords[2], coords[3], outline=color, width=3)

            # Don't auto-close the overlay - let it stay until next screen change or manual close
            # The overlay will be closed by the main loop when appropriate
            
            # Update the display without blocking
            root.update()
            print("Overlay displayed successfully - will stay until next screen activity")
            
        except Exception as e:
            logging.error(f"Failed to create Tkinter overlay: {e}")
            print(f"Failed to create overlay: {e}")

    def _close_overlay(self):
        """Close the current overlay if it exists."""
        if self.overlay_root:
            try:
                self.overlay_root.destroy()
                self.overlay_root = None
            except:
                pass

    def _handle_all_fields_matched(self):
        """Handle when all fields match - send configured key press if enabled."""
        try:
            # Check if key press is configured and enabled
            automation_config = self.config.get("automation", {})
            if not automation_config.get("send_key_on_all_match", False):
                return
                
            key_to_send = automation_config.get("key_on_all_match", "f12")
            delay_seconds = automation_config.get("key_delay_seconds", 0.5)
            
            print(f"🎉 All fields matched! Sending '{key_to_send}' key press in {delay_seconds}s...")
            
            # Wait the configured delay before sending the key
            time.sleep(delay_seconds)
            
            # Send the key press using pyautogui
            pyautogui.press(key_to_send.lower())
            print(f"✅ Sent '{key_to_send}' key press successfully")
            
        except Exception as e:
            logging.error(f"Error sending key press: {e}")
            print(f"❌ Failed to send key press: {e}")

    def _extract_rx_number(self, trigger_text: str) -> str:
        """Extract the Rx number from the trigger text like 'Pre-Check Rx - 97421'."""
        import re
        try:
            # Look for pattern like "Rx - 12345" or "Rx-12345" or just numbers after "rx"
            # This should match various formats like:
            # "Pre-Check Rx - 97421"
            # "Pre-Check Rx-97421" 
            # "Pre-check rx 97421"
            patterns = [
                r'rx\s*-\s*(\d+)',  # "rx - 12345" or "rx-12345"
                r'rx\s+(\d+)',      # "rx 12345"
                r'(\d{4,})',        # Any 4+ digit number as fallback
            ]
            
            text_lower = trigger_text.lower()
            for pattern in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    return match.group(1)
            
            return ""
        except Exception as e:
            logging.error(f"Error extracting Rx number from '{trigger_text}': {e}")
            return ""

    def _check_trigger(self, screenshot: Image.Image) -> Tuple[bool, str]:
        """Check if the trigger text is present and return both trigger status and Rx number."""
        try:
            trigger_region = tuple(self.config["regions"]["trigger"])
            trigger_text = self._get_text_from_region(screenshot, trigger_region)
            
            # Check if any variation of "pre-check rx" is present
            trigger_words = ["pre", "check", "rx"]
            text_lower = trigger_text.lower()
            
            # Use rapidfuzz to find similar words
            found_words = []
            words_in_text = text_lower.split()
            
            for target_word in trigger_words:
                for word in words_in_text:
                    if fuzz.ratio(word, target_word) >= 70:  # 70% similarity threshold
                        found_words.append(target_word)
                        break
            
            trigger_detected = len(found_words) >= 2  # Need at least 2 of the 3 words
            
            # Extract Rx number if trigger is detected
            rx_number = ""
            if trigger_detected:
                rx_number = self._extract_rx_number(trigger_text)
                
            return trigger_detected, rx_number
        except Exception as e:
            logging.error(f"Error checking trigger: {e}")
            return False, ""

    def _verify_all_fields(self, screenshot: Image.Image):
        """Run verification on all fields and show overlay."""
        if self.verification_in_progress:
            print("Verification already in progress, skipping...")
            return
            
        try:
            self.verification_in_progress = True
            print("Running field verification...")
            results = self.verify_fields(screenshot)
            
            # Log the results with Rx number and rotation
            self._log_rx_verification(self.last_rx_number or "", results)
            
            # Log the results summary to console
            matches = sum(1 for r in results.values() if r["match"])
            total = len(results)
            print(f"Verification complete: {matches}/{total} fields matched")
            
            # Check if all fields matched and send key press if configured
            if matches == total and total > 0:
                self._handle_all_fields_matched()
            
            self._show_tk_overlay(results)
        except Exception as e:
            logging.error(f"Error during verification: {e}")
            print(f"Error during verification: {e}")
        finally:
            self.verification_in_progress = False

    def stop(self):
        """Stop the monitoring loop gracefully"""
        print("Stop requested - monitoring will terminate after current iteration...")
        self.should_stop = True
        self._close_overlay()  # Clean up any active overlay

    def run(self):
        """Main loop to monitor for the trigger text with improved responsiveness."""
        print("Starting to monitor for 'pre-check rx' text...")
        print("Using adaptive timing for better responsiveness...")
        print("Press Ctrl+C to stop")
        
        consecutive_no_change = 0
        last_trigger_check = 0
        last_hash_check = 0
        
        while True:
            try:
                # Check if we should stop the monitoring
                if self.should_stop:
                    print("Stopping monitoring as requested...")
                    self._close_overlay()  # Clean up overlay before stopping
                    break
                    
                current_time = time.time()
                screenshot = pyautogui.screenshot()
                
                # Only check for screen changes every few iterations to reduce overhead
                should_check_hash = (current_time - last_hash_check) >= 1.0  # Check hash every 1 second max
                screen_changed = False
                
                if should_check_hash:
                    screen_changed = self._has_screen_changed(screenshot)
                    last_hash_check = current_time
                    
                    if not screen_changed:
                        consecutive_no_change += 1
                    else:
                        consecutive_no_change = 0
                        print("Screen activity detected")
                
                # Close overlay if it's been shown for too long (safety timeout)
                if self.overlay_root is not None:
                    overlay_age = current_time - self.overlay_created_time
                    max_overlay_time = 30.0  # Maximum 30 seconds
                    if overlay_age >= max_overlay_time:
                        print(f"Overlay timeout reached ({overlay_age:.1f}s) - closing overlay")
                        self._close_overlay()
                
                # Only close overlay when screen changes if it's been displayed for at least 3 seconds
                # OR if user has navigated away from the trigger area
                if screen_changed and self.overlay_root is not None:
                    overlay_age = current_time - self.overlay_created_time
                    if overlay_age >= 3.0:  # Give overlay at least 3 seconds to be seen
                        print("Screen changed and overlay is old enough - closing overlay")
                        self._close_overlay()
                    else:
                        print(f"Screen changed but overlay is too new ({overlay_age:.1f}s), keeping it visible")
                
                # Much more aggressive adaptive timing
                if consecutive_no_change == 0:
                    # Screen is changing - use fast polling
                    sleep_time = self.config["timing"]["fast_polling_seconds"]
                elif consecutive_no_change < 5:
                    # Recently changed - medium polling
                    sleep_time = self.config["timing"]["fast_polling_seconds"] * 2
                elif consecutive_no_change < 15:
                    # Been static for a while - slow polling
                    sleep_time = self.config["timing"]["fast_polling_seconds"] * 4
                else:
                    # Been static for a long time - very slow polling
                    sleep_time = min(self.config["timing"]["max_static_sleep_seconds"], 2.0)
                
                # Only check trigger on screen changes, or at much longer intervals when static
                trigger_interval = self.config["timing"]["trigger_check_interval_seconds"]
                if consecutive_no_change > 10:
                    # When screen is very static, check trigger much less frequently
                    trigger_interval *= 3
                    
                should_check_trigger = (screen_changed or 
                                      (current_time - last_trigger_check) >= trigger_interval)
                
                trigger_detected = False
                current_rx_number = ""
                if should_check_trigger:
                    trigger_detected, current_rx_number = self._check_trigger(screenshot)
                    last_trigger_check = current_time
                
                if trigger_detected:
                    # Clear the trigger lost timer since we found the trigger again
                    if hasattr(self, 'trigger_lost_time') and self.trigger_lost_time is not None:
                        self.trigger_lost_time = None
                        
                    if current_rx_number:
                        print(f"Detected trigger text: 'pre-check rx' for Rx #{current_rx_number}")
                    else:
                        print("Detected trigger text: 'pre-check rx' (no Rx number found)")
                    consecutive_no_change = 0  # Reset since we found something important
                    
                    # Check if this is a new prescription using Rx number FIRST
                    is_new_prescription = False
                    
                    if current_rx_number:
                        # We have an Rx number - compare with last one
                        if not self.recently_triggered:
                            # First time detecting any trigger
                            is_new_prescription = True
                        elif self.last_rx_number is None:
                            # We had a trigger before but no Rx number, now we have one
                            is_new_prescription = True
                        elif current_rx_number != self.last_rx_number:
                            # Different Rx number from last time
                            is_new_prescription = True
                        # else: same Rx number, not new
                    else:
                        # No Rx number found - fall back to old behavior only if we haven't triggered recently
                        is_new_prescription = not self.recently_triggered
                    
                    # Only apply cooldown if this is the SAME prescription
                    if not is_new_prescription:
                        # Check cooldown to prevent rapid re-triggering of the same Rx
                        time_since_last_trigger = current_time - self.last_trigger_time
                        min_cooldown = self.config["timing"]["same_prescription_wait_seconds"] * 2  # Double the wait time for cooldown
                        
                        if time_since_last_trigger < min_cooldown:
                            print(f"Same Rx detected in cooldown period ({time_since_last_trigger:.1f}s < {min_cooldown:.1f}s)")
                            time.sleep(sleep_time)
                            continue
                    
                    if is_new_prescription:
                        if current_rx_number and self.last_rx_number:
                            print(f"New Rx detected: #{current_rx_number} (was #{self.last_rx_number}), starting verification...")
                        elif current_rx_number:
                            print(f"New Rx detected: #{current_rx_number}, starting verification...")
                        else:
                            print("New prescription detected (no Rx number), starting verification...")
                            
                        self.last_rx_number = current_rx_number
                        self.recently_triggered = True
                        self.last_trigger_time = current_time
                        
                        # Configurable wait time for better responsiveness
                        verification_wait = self.config["timing"]["verification_wait_seconds"]
                        time.sleep(verification_wait)
                        
                        # Take a fresh screenshot after the wait
                        fresh_screenshot = pyautogui.screenshot()
                        print("About to verify fields and show overlay...")
                        self._verify_all_fields(fresh_screenshot)
                    else:
                        # Still on same prescription - set a longer cooldown to prevent immediate re-triggering
                        if current_rx_number:
                            print(f"Same Rx as before (#{current_rx_number}), entering cooldown period...")
                        else:
                            print("Same prescription as before, entering cooldown period...")
                        self.last_trigger_time = current_time
                        same_rx_wait = self.config["timing"]["same_prescription_wait_seconds"]
                        time.sleep(same_rx_wait)
                else:
                    # Only reset when trigger is no longer detected AND we've been away for a while
                    if self.recently_triggered:
                        # Start counting how long we've been away from the trigger
                        if not hasattr(self, 'trigger_lost_time') or self.trigger_lost_time is None:
                            self.trigger_lost_time = current_time
                            print("Trigger text no longer detected, starting reset timer...")
                        else:
                            # Check if we've been away long enough to consider it a real navigation away
                            time_since_lost = current_time - self.trigger_lost_time
                            reset_delay = 5.0  # Wait 5 seconds before resetting
                            
                            if time_since_lost >= reset_delay:
                                print(f"Trigger text absent for {time_since_lost:.1f}s, resetting for next prescription")
                                self.recently_triggered = False
                                self.last_prescription_data = None
                                self.last_rx_number = None  # Reset Rx number
                                self.last_trigger_time = 0  # Reset trigger time
                                self.overlay_created_time = 0  # Reset overlay timing
                                self.trigger_lost_time = None  # Reset the lost timer
                                # Close overlay when leaving the trigger area
                                self._close_overlay()
                            # else: still within the reset delay, don't reset yet
                    else:
                        # Not recently triggered, make sure lost timer is cleared
                        if hasattr(self, 'trigger_lost_time'):
                            self.trigger_lost_time = None
                
                # Show current status every 30 seconds when static
                if consecutive_no_change > 0 and consecutive_no_change % 30 == 0:
                    print(f"System idle - polling every {sleep_time:.1f}s (no changes for {consecutive_no_change} checks)")
                
                # Adaptive sleep based on activity
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                # Clean up overlay on exit
                self._close_overlay()
                break
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
        return None
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
