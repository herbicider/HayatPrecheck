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
        
        # Cache abbreviations for efficiency (load once at startup)
        self._abbreviations = self._load_abbreviations()
        logging.info(f"Cached {len(self._abbreviations)} abbreviations for efficient matching")
        
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

    def _load_abbreviations(self) -> Dict[str, str]:
        """
        Loads pharmaceutical abbreviations from external JSON file for easy maintenance.
        This method is called once during initialization to avoid repeated file I/O.
        
        The abbreviations are loaded from 'abbreviations.json' in the same directory.
        If the file doesn't exist, returns an empty dictionary and logs a warning.
        
        The JSON file format should be:
        {
            " abbr ": " full expansion ",
            " another_abbr ": " another expansion "
        }
        """
        abbreviations = {}
        
        # Load abbreviations from external JSON file
        try:
            abbrev_file = os.path.join(os.path.dirname(__file__), "abbreviations.json")
            if os.path.exists(abbrev_file):
                with open(abbrev_file, 'r', encoding='utf-8') as f:
                    loaded_abbreviations = json.load(f)
                    # Filter out comment keys (they start with "//")
                    abbreviations = {k: v for k, v in loaded_abbreviations.items() if not k.startswith("//")}
                    logging.info(f"Loaded {len(abbreviations)} abbreviations from {abbrev_file}")
            else:
                logging.warning(f"Abbreviations file not found: {abbrev_file}")
                logging.warning("Text cleaning will work with limited abbreviation support")
        except Exception as e:
            logging.error(f"Could not load abbreviations from file: {e}")
            logging.warning("Text cleaning will work with limited abbreviation support")
        
        return abbreviations
        # Default comprehensive abbreviations



    def reload_abbreviations(self):
        """
        Reload abbreviations from file without restarting the application.
        Useful for testing new abbreviations or when the JSON file is updated.
        """
        try:
            self._abbreviations = self._load_abbreviations()
            logging.info("Abbreviations reloaded successfully")
        except Exception as e:
            logging.error(f"Error reloading abbreviations: {e}")

    def get_loaded_abbreviations_count(self) -> int:
        """Return the number of loaded abbreviations for debugging."""
        return len(self._abbreviations)

    def _clean_drug_name(self, text: str) -> str:
        """
        Specialized cleaning for drug names with enhanced semantic matching.
        PRESERVES dosage information since it's critical for identification.
        Different dosages are considered different drugs (metformin 500mg ≠ metformin 1000mg).
        Equivalent salt forms are normalized (metformin HCl = metformin hydrochloride).
        Different salt forms are maintained as distinct (metoprolol tartrate ≠ metoprolol succinate).
        """
        if not text:
            return ""
        
        # Start with basic cleaning but preserve numbers and units
        text = text.lower().strip()
        
        # Remove brand names in parentheses (e.g., "Metformin (Glucophage)" -> "Metformin")
        import re
        text = re.sub(r'\s*\([^)]*\)', '', text)
        
        # Get comprehensive pharmaceutical abbreviations (cached at startup)
        abbreviations = self._abbreviations
        
        # Apply abbreviation expansions
        # Add spaces around text to ensure word boundary matching
        text_with_spaces = f" {text} "
        for abbr, full in abbreviations.items():
            text_with_spaces = text_with_spaces.replace(abbr, full)
        
        # Remove the extra spaces we added
        text = text_with_spaces.strip()
        
        # Remove most punctuation BUT preserve numbers, decimal points in doses, and % signs
        # This allows "10.5mg" and "2.5%" to remain intact
        for char in string.punctuation:
            if char not in ['.', '%']:  # Keep dots for decimals and % for percentages
                text = text.replace(char, ' ')
        
        # Clean up extra spaces
        text = ' '.join(text.split())
        
        # Normalize equivalent salt forms - but keep distinct salts separate
        # Only normalize specific cases where the salts are pharmacologically equivalent
        equivalent_salt_forms = {
            'hydrochloride': 'hcl',
            'hydrobromide': 'hbr',
            'hcl': 'hcl',           # Already normalized, keep it
            'hydrochloride': 'hcl',  # Normalize this variant
            'hydrochlor': 'hcl',    # Normalize this abbreviation too
        }
        
        # Apply equivalent salt normalization carefully
        for full_salt, abbr_salt in equivalent_salt_forms.items():
            # Only replace exact salt form matches surrounded by spaces
            text = re.sub(rf'\b{full_salt}\b', abbr_salt, text)
        
        # Do NOT normalize distinct salts like tartrate/succinate since they're different drugs
        
        # Standardize spacing in dose information
        # This ensures "metformin500mg" becomes "metformin 500mg" for better matching
        text = re.sub(r'(\d+)([a-z]+)', r'\1 \2', text)  # "500mg" -> "500 mg"
        text = re.sub(r'([a-z])(\d+)', r'\1 \2', text)   # "metformin500" -> "metformin 500"
        
        # Final spacing cleanup
        text = ' '.join(text.split())
        
        return text

    def _enhanced_drug_name_match(self, entered_text: str, source_text: str, threshold: int = 80) -> Tuple[float, bool]:
        """
        Enhanced drug name matching with semantic awareness.
        Considers dosage information critical - different dosages are different drugs.
        Different salt forms (tartrate vs succinate) are also considered different drugs.
        Requires strong similarity in the actual drug name, not just dosage/form matching.
        
        Args:
            entered_text: Text from the entered field
            source_text: Text from the source field  
            threshold: Similarity threshold for matching (0-100)
            
        Returns:
            Tuple of (score, is_match) where score is 0-100 and is_match is boolean
        """
        # Clean both texts with drug-specific cleaning that preserves dosages
        clean_entered = self._clean_drug_name(entered_text)
        clean_source = self._clean_drug_name(source_text)
        
        if not clean_entered or not clean_source:
            return 0.0, False
        
        # Check if they have different dosages by extracting and comparing dose information
        import re
        
        # Extract dosage information from both strings
        entered_dosages = re.findall(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', clean_entered)
        source_dosages = re.findall(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', clean_source)
        
        # If both have dosages but they're different, consider it a mismatch
        if entered_dosages and source_dosages:
            # Normalize dosages for comparison (convert to same format)
            def normalize_dosage(dosages):
                normalized = []
                for dose_value, unit in dosages:
                    # Convert to float for numeric comparison
                    dose_num = float(dose_value)
                    normalized.append((dose_num, unit.lower()))
                return normalized
            
            entered_norm = normalize_dosage(entered_dosages)
            source_norm = normalize_dosage(source_dosages)
            
            if entered_norm != source_norm:
                print(f"Dosage mismatch detected: {entered_norm} vs {source_norm}")
                return 50.0, False  # Return a mediocre score to indicate partial match but dosage difference
        
        # Check for different salt forms that should be considered different drugs
        distinct_salt_pairs = [
            ('tartrate', 'succinate'),  # metoprolol tartrate vs succinate are different
            ('ir', 'er'),               # immediate release vs extended release
            ('ir', 'sr'),               # immediate release vs sustained release
            ('ir', 'xr'),               # immediate release vs extended release
            ('er', 'sr'),               # different extended release formulations
            ('er', 'xr'),               # different extended release formulations
        ]
        
        # Check if they have different distinct salt forms
        for salt1, salt2 in distinct_salt_pairs:
            if ((salt1 in clean_entered and salt2 in clean_source) or 
                (salt2 in clean_entered and salt1 in clean_source)):
                print(f"Different salt forms detected: {salt1} vs {salt2}")
                return 40.0, False  # Return a low score to indicate different drugs
        
        # Extract the drug name portion (remove dosage and common terms to focus on drug name)
        def extract_drug_name_portion(text):
            # Remove dosage information and common pharmaceutical terms
            drug_text = text
            # Remove dosages
            drug_text = re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', '', drug_text)
            # Remove common pharmaceutical terms
            common_terms = ['tablet', 'tablets', 'capsule', 'capsules', 'by mouth', 'oral', 
                          'extended release', 'immediate release', 'delayed release',
                          'milligram', 'microgram', 'gram', 'milliliter', 'liter']
            for term in common_terms:
                drug_text = drug_text.replace(term, '')
            # Clean up extra spaces
            drug_text = ' '.join(drug_text.split())
            return drug_text.strip()
        
        # Extract just the drug name portions
        entered_drug_name = extract_drug_name_portion(clean_entered)
        source_drug_name = extract_drug_name_portion(clean_source)
        
        print(f"Drug name portions - Entered: '{entered_drug_name}', Source: '{source_drug_name}'")
        
        # Get the first word (the actual drug name) from each string
        entered_first_word = entered_drug_name.split()[0] if entered_drug_name else ""
        source_first_word = source_drug_name.split()[0] if source_drug_name else ""
        
        print(f"First words (main drug names) - Entered: '{entered_first_word}', Source: '{source_first_word}'")
        
        # Initialize first_word_similarity for later use
        first_word_similarity = 0
        
        # Check for brand name in parentheses in the original source text
        # This handles cases like "empagliflozin 10mg (JARDIANCE)" where the brand is in parentheses
        brand_name_in_source = ""
        import re
        brand_match = re.search(r'\(([^)]+)\)', source_text.lower())
        if brand_match:
            # Extract just the first word of what's in parentheses (the brand name)
            brand_in_parentheses = brand_match.group(1).strip()
            # Remove common words and get the first significant word
            brand_words = [w for w in brand_in_parentheses.split() if w not in ['tablet', 'capsule', 'mg', 'milligram']]
            brand_name_in_source = brand_words[0] if brand_words else ""
            print(f"Brand name found in source parentheses: '{brand_name_in_source}'")
        
        # If the first words (main drug names) are too different, check against brand name too
        if entered_first_word and source_first_word:
            first_word_similarity = fuzz.ratio(entered_first_word, source_first_word)
            print(f"First word similarity: {first_word_similarity:.2f}")
            
            # If direct comparison fails, try comparing with brand name in parentheses
            brand_similarity = 0
            if brand_name_in_source and first_word_similarity < 70:
                brand_similarity = fuzz.ratio(entered_first_word, brand_name_in_source)
                print(f"Brand name similarity ('{entered_first_word}' vs '{brand_name_in_source}'): {brand_similarity:.2f}")
            
            # Use the better of the two similarities
            best_first_word_similarity = max(first_word_similarity, brand_similarity)
            
            # If both the generic and brand name comparisons fail, this is likely different drugs
            if best_first_word_similarity < 70:
                print(f"Both generic and brand name comparisons failed ({best_first_word_similarity:.2f}%), likely different drugs")
                # Return a low score that won't pass most thresholds
                return min(60.0, best_first_word_similarity + 15), False
            elif brand_similarity > first_word_similarity:
                print(f"Brand name match found! Using brand similarity: {brand_similarity:.2f}")
                # Update first_word_similarity to use the better brand match
                first_word_similarity = brand_similarity
        
        # If drug names are too different overall, don't allow high scores based on dosage/form matching
        if entered_drug_name and source_drug_name:
            drug_name_similarity = fuzz.ratio(entered_drug_name, source_drug_name)
            print(f"Full drug name similarity: {drug_name_similarity:.2f}")
            
            # If drug names are very different (< 60% similar), cap the overall score
            if drug_name_similarity < 60:
                print(f"Drug names too different ({drug_name_similarity:.2f}%), capping score to prevent false positive")
                # Return a moderate score that won't pass most thresholds
                return min(65.0, drug_name_similarity + 10), False
        
        # Try multiple matching strategies and take the best score
        scores = []
        
        # 1. Direct fuzzy matching - most important for exact matches
        scores.append(fuzz.ratio(clean_entered, clean_source) * 1.2)  # Weighted higher
        
        # 2. Token sort ratio (handles word order differences)
        scores.append(fuzz.token_sort_ratio(clean_entered, clean_source))
        
        # 3. Token set ratio (handles extra words) - weighted lower to avoid false matches
        scores.append(fuzz.token_set_ratio(clean_entered, clean_source) * 0.8)
        
        # 4. Partial ratio (handles one string being a subset of another) - weighted lower
        scores.append(fuzz.partial_ratio(clean_entered, clean_source) * 0.7)
        
        # Take the best score but cap at 100
        best_score = min(100, max(scores))
        
        # Additional check: if drug names are different but dosage/form match,
        # reduce the score significantly to prevent false positives
        # Give extra weight to the first word (main drug name) similarity
        if entered_drug_name and source_drug_name:
            drug_name_similarity = fuzz.ratio(entered_drug_name, source_drug_name)
            # Use the first_word_similarity that was calculated above (includes brand name check)
            
            # If either the first words or overall drug names don't match well, adjust the score
            if (first_word_similarity < 70 or drug_name_similarity < 70) and best_score > 75:
                # Weight the first word similarity more heavily since it's the actual drug name
                primary_similarity = (first_word_similarity * 0.7) + (drug_name_similarity * 0.3)
                # Significantly reduce score when drug names don't match well
                adjusted_score = (best_score * 0.5) + (primary_similarity * 0.5)
                print(f"Adjusted score due to drug name mismatch:")
                print(f"  First word similarity (incl. brand check): {first_word_similarity:.2f}")
                print(f"  Overall drug similarity: {drug_name_similarity:.2f}")
                print(f"  Primary similarity (weighted): {primary_similarity:.2f}")
                print(f"  Final adjustment: {best_score:.2f} -> {adjusted_score:.2f}")
                best_score = adjusted_score
        
        is_match = best_score >= threshold
        
        return best_score, is_match

    def _calculate_drug_name_score(self, entered_text: str, source_text: str) -> float:
        """Enhanced drug name matching with multiple strategies and better validation."""
        if not entered_text or not source_text:
            return 0.0
        
        # Clean and normalize both texts
        entered_clean = self._clean_text(entered_text).lower().strip()
        source_clean = self._clean_text(source_text).lower().strip()
        
        if not entered_clean or not source_clean:
            return 0.0
        
        # Strategy 1: Direct fuzzy matching
        direct_score = fuzz.ratio(entered_clean, source_clean)
        
        # Strategy 2: Token sort ratio (handles word order differences)
        token_sort_score = fuzz.token_sort_ratio(entered_clean, source_clean)
        
        # Strategy 3: Token set ratio (handles extra words) - but be more careful
        token_set_score = fuzz.token_set_ratio(entered_clean, source_clean)
        
        # Strategy 4: Partial ratio (handles subsets)
        partial_score = fuzz.partial_ratio(entered_clean, source_clean)
        
        # Strategy 5: Core drug name matching (first significant word)
        entered_words = [w for w in entered_clean.split() if len(w) > 2]
        source_words = [w for w in source_clean.split() if len(w) > 2]
        
        core_score = 0
        if entered_words and source_words:
            core_score = fuzz.ratio(entered_words[0], source_words[0])
        
        # Strategy 6: Cross-word matching (any word to any word) - but require multiple word matches
        cross_score = 0
        if entered_words and source_words:
            word_matches = []
            for e_word in entered_words:
                for s_word in source_words:
                    word_score = fuzz.ratio(e_word, s_word)
                    if word_score > 80:  # Only consider strong word matches
                        word_matches.append(word_score)
            
            # Require at least 2 strong word matches or 1 perfect match for high scores
            if len(word_matches) >= 2:
                cross_score = sum(word_matches) / len(word_matches)
            elif word_matches and max(word_matches) >= 95:
                cross_score = max(word_matches)
            else:
                cross_score = max(word_matches) if word_matches else 0
                # Penalize single word matches
                if len(word_matches) == 1 and cross_score < 95:
                    cross_score = min(cross_score, 75)
        
        # Use weighted average instead of just max to prevent single-word false positives
        scores = [direct_score, token_sort_score, partial_score, core_score]
        
        # Only include token_set_score and cross_score if they're not outliers
        if token_set_score <= max(scores) + 20:  # Not too much higher than other scores
            scores.append(token_set_score)
        
        if cross_score <= max(scores) + 20:  # Not too much higher than other scores
            scores.append(cross_score)
        
        # Use weighted average favoring direct and token sort scores
        final_score = (direct_score * 2 + token_sort_score * 2 + sum(scores)) / (4 + len(scores))
        
        return final_score

    def _clean_text(self, text: str) -> str:
        """
        Normalizes text by lowercasing, expanding abbreviations, and removing punctuation.
        Uses cached pharmaceutical abbreviations for efficient processing.
        """
        if not text:
            return ""
        text = text.lower().strip()
        
        # Remove brand names in parentheses (e.g., "Metformin (Glucophage)" -> "Metformin")
        import re
        text = re.sub(r'\s*\([^)]*\)', '', text)
        
        # Get comprehensive pharmaceutical abbreviations (cached at startup)
        abbreviations = self._abbreviations
        
        # Apply abbreviation expansions
        # Add spaces around text to ensure word boundary matching
        text_with_spaces = f" {text} "
        for abbr, full in abbreviations.items():
            text_with_spaces = text_with_spaces.replace(abbr, full)
        
        # Remove the extra spaces we added
        text = text_with_spaces.strip()
        
        # Remove punctuation and consolidate whitespace
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text

    def _normalize_name(self, name: str, is_entered_field: bool = False) -> str:
        """
        Normalizes names to handle format differences and medical titles.
        PRESERVES the original name format and order - 'John Smith' is NOT the same as 'Smith John'.
        
        Args:
            name: The name to normalize
            is_entered_field: True if this is from the left side (entered field), 
                            False if from right side (source field)
        
        Returns:
            Normalized name preserving the original format and order
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
            'CRNA', 'C.R.N.A.', 'APRN', 'A.P.R.N.', 'DPM', 'D.P.M.', 'PMHNP', 'P.M.H.N.P.',
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
        
        # Remove common suffixes like "of", "jr", "sr", etc. but preserve the overall format
        clean_name = re.sub(r'\s+(of|jr|sr|iii|ii|iv)$', '', clean_name, flags=re.IGNORECASE)
        
        # Handle OCR artifacts but maintain original formatting
        clean_name = clean_name.replace('|', ' ')  # Remove pipe separators
        clean_name = clean_name.replace('_', ' ')  # Remove underscores (OCR artifacts)
        clean_name = ' '.join(clean_name.split())  # Clean up extra spaces
        
        # Handle multiple prescribers separated by "/"
        if '/' in clean_name:
            prescribers = clean_name.split('/')
            processed_prescribers = []
            
            for prescriber in prescribers:
                prescriber = prescriber.strip()
                if prescriber:
                    # Apply common cleaning but preserve the format
                    processed_prescribers.append(prescriber.lower())
            
            # Return all prescribers joined by " / " for later processing
            if processed_prescribers:
                return " / ".join(processed_prescribers)
        
        # Return the cleaned name in lowercase but preserve the original format
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
            elif field_name == "drug_name":
                # Use enhanced drug name cleaning for drug fields
                cleaned_entered = self._clean_drug_name(entered_text)
                cleaned_source = self._clean_drug_name(source_text)
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
                threshold_key = config["threshold_key"]
                threshold = thresholds[threshold_key]
                
                # Special handling for drug names with enhanced semantic matching
                if field_name == "drug_name":
                    score, is_match = self._enhanced_drug_name_match(
                        cleaned_entered, cleaned_source, threshold
                    )
                # Special handling for prescriber names - check multiple prescribers separated by "/"
                elif field_name == "prescriber_name" and " / " in cleaned_source:
                    scorer = getattr(fuzz, config["score_fn"], fuzz.ratio)
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
                    scorer = getattr(fuzz, config["score_fn"], fuzz.ratio)
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
