import re
import string
import json
import os
import logging
from typing import Dict, Any, Tuple
from rapidfuzz import fuzz

from core.logger_config import log_field_details

class ComparisonEngine:
    """Handles text cleaning, normalization, and comparison logic."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.advanced_settings = config.get("advanced_settings", {})
        self._abbreviations = self._load_abbreviations()
        logging.info(f"Cached {len(self._abbreviations)} abbreviations for efficient matching")

    def _load_abbreviations(self) -> Dict[str, str]:
        """Loads pharmaceutical abbreviations from config/abbreviations.json (preferred), with fallbacks."""
        try:
            candidates = []

            # 1) Explicit path from config (preferred override)
            explicit = (
                self.config.get("abbreviations_path")
                or self.config.get("paths", {}).get("abbreviations")
                or self.config.get("paths", {}).get("abbreviations_path")
            )
            if explicit:
                candidates.append(explicit)

            # 2) Default new location: <project_root>/config/abbreviations.json
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
            candidates.append(os.path.join(project_root, "config", "abbreviations.json"))

            # 3) Legacy fallback: alongside this file (<project_root>/core/abbreviations.json)
            candidates.append(os.path.join(os.path.dirname(__file__), "abbreviations.json"))

            for abbrev_file in candidates:
                if not abbrev_file:
                    continue
                if os.path.exists(abbrev_file):
                    with open(abbrev_file, 'r', encoding='utf-8') as f:
                        loaded_abbreviations = json.load(f)
                        cleaned = {k: v for k, v in loaded_abbreviations.items() if not str(k).startswith("//")}
                        logging.info(f"Loaded abbreviations from: {abbrev_file} ({len(cleaned)} entries)")
                        return cleaned
                else:
                    logging.debug(f"Abbreviations file not found at: {abbrev_file}")
        except Exception as e:
            logging.error(f"Could not load abbreviations from file: {e}")
        
        logging.warning("Text cleaning will work with limited abbreviation support")
        return {}

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
        
        if len(entered_text.strip()) < 3 or len(source_text.strip()) < 3:
            logging.debug(f"Text too short for reliable matching: '{entered_text}' vs '{source_text}'")
            return 0.0, False
        
        clean_entered = self._clean_drug_name(entered_text)
        clean_source = self._clean_drug_name(source_text)
        
        if not clean_entered or not clean_source or len(clean_entered) < 2 or len(clean_source) < 2:
            logging.debug(f"Cleaned text too short or empty: '{clean_entered}' vs '{clean_source}'")
            return 0.0, False
        
        entered_dosages = re.findall(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', clean_entered)
        source_dosages = re.findall(r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', clean_source)
        if entered_dosages and source_dosages:
            normalize_dosage = lambda dosages: sorted([(float(v), u.lower()) for v, u in dosages])
            if normalize_dosage(entered_dosages) != normalize_dosage(source_dosages):
                logging.debug(f"Dosage mismatch detected: {entered_dosages} vs {source_dosages}")
                return 50.0, False

        distinct_salt_pairs = self.advanced_settings.get("matching", {}).get("distinct_salt_pairs", [])
        for salt1, salt2 in distinct_salt_pairs:
            if (salt1 in clean_entered and salt2 in clean_source) or \
               (salt2 in clean_entered and salt1 in clean_source):
                logging.debug(f"Different salt forms detected: {salt1} vs {salt2}")
                return 40.0, False

        def extract_drug_name(t):
            for term in ['tablet', 'capsule', 'oral', 'extended release', 'etc']:
                t = t.replace(term, '')
            return ' '.join(re.sub(r'\b\d+(\.\d+)?\s*(mg|mcg|ug|g|ml|l|units?|iu|%)\b', '', t).split())
        
        entered_drug_name = extract_drug_name(clean_entered)
        source_drug_name = extract_drug_name(clean_source)
        logging.debug(f"Drug name portions - Entered: '{entered_drug_name}', Source: '{source_drug_name}'")

        scores = [
            fuzz.ratio(clean_entered, clean_source) * 1.2,
            fuzz.token_sort_ratio(clean_entered, clean_source),
            fuzz.token_set_ratio(clean_entered, clean_source) * 0.8,
            fuzz.partial_ratio(clean_entered, clean_source) * 0.7
        ]
        best_score = min(100, max(scores))
        
        is_match = best_score >= threshold
        return best_score, is_match

    def _enhanced_name_match(self, entered_text: str, source_text: str, threshold: float) -> Tuple[float, bool]:
        """Enhanced name matching that's more tolerant of OCR errors."""
        
        clean_entered = self._normalize_name(entered_text, is_entered_field=True)
        clean_source = self._normalize_name(source_text, is_entered_field=False)
        
        if not clean_entered or not clean_source:
            return 0.0, False
        
        ratio_score = fuzz.ratio(clean_entered, clean_source)
        token_sort_score = fuzz.token_sort_ratio(clean_entered, clean_source)
        token_set_score = fuzz.token_set_ratio(clean_entered, clean_source)
        partial_score = fuzz.partial_ratio(clean_entered, clean_source)
        
        def single_char_diff_bonus(s1: str, s2: str) -> float:
            if abs(len(s1) - len(s2)) <= 1:
                import difflib
                diff_count = sum(1 for op in difflib.ndiff(s1, s2) if op.startswith('- ') or op.startswith('+ '))
                if diff_count <= 2:
                    return 20.0
            return 0.0
        
        ocr_bonus = single_char_diff_bonus(clean_entered, clean_source)
        
        best_score = max(ratio_score, token_sort_score, token_set_score, partial_score) + ocr_bonus
        best_score = min(100.0, best_score)
        
        is_match = best_score >= threshold
        
        if ocr_bonus > 0:
            logging.debug(f"OCR bonus applied: {ocr_bonus} points for likely single-char error")
        
        return best_score, is_match

    def _match_prescriber_names(self, entered_text: str, source_text: str, threshold: float) -> Tuple[float, bool]:
        """Special matching logic for prescriber names that may contain multiple names separated by '/'."""
        
        clean_entered = self._normalize_name(entered_text, is_entered_field=True)
        
        if '/' in source_text:
            source_prescribers = [p.strip() for p in source_text.split('/') if p.strip()]
            
            best_score = 0.0
            best_prescriber = ""
            for prescriber in source_prescribers:
                clean_prescriber = self._normalize_name(prescriber, is_entered_field=False)
                if not clean_prescriber:
                    continue
                    
                ratio_score = fuzz.ratio(clean_entered, clean_prescriber)
                token_sort_score = fuzz.token_sort_ratio(clean_entered, clean_prescriber)
                token_set_score = fuzz.token_set_ratio(clean_entered, clean_prescriber)
                
                prescriber_score = max(ratio_score, token_sort_score, token_set_score)
                
                if prescriber_score > best_score:
                    best_score = prescriber_score
                    best_prescriber = clean_prescriber
            
            logging.debug(f"Best prescriber match: '{clean_entered}' vs '{best_prescriber}' = {best_score:.2f}")
            is_match = best_score >= threshold
            return best_score, is_match
        else:
            clean_source = self._normalize_name(source_text, is_entered_field=False)
            score = max(
                fuzz.ratio(clean_entered, clean_source),
                fuzz.token_sort_ratio(clean_entered, clean_source),
                fuzz.token_set_ratio(clean_entered, clean_source)
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
        
        text = re.sub(r'\s*\(([^)]*)\)\s*', r' \1 ', text)
        
        text_with_spaces = f" {text} "
        for abbr, full in self._abbreviations.items():
            text_with_spaces = text_with_spaces.replace(abbr, full)
        
        text = text_with_spaces.strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())

    def _clean_phone_number(self, text: str) -> str:
        """Cleans phone number to a consistent format of digits only."""
        if not text:
            return ""
        return re.sub(r'\\D', '', text)

    def _normalize_address(self, text: str) -> str:
        """Normalizes address to focus on street number and name."""
        if not text:
            return ""
        
        text = text.lower().strip()
        
        # Common address abbreviations
        address_abbreviations = {
            r'\\bn\\b': 'north', r'\\bs\\b': 'south', r'\\be\\b': 'east', r'\\bw\\b': 'west',
            r'\\bst\\b': 'street', r'\\bave\\b': 'avenue', r'\\brd\\b': 'road', r'\\bln\\b': 'lane',
            r'\\bdr\\b': 'drive', r'\\bblvd\\b': 'boulevard', r'\\bct\\b': 'court'
        }
        for abbr, full in address_abbreviations.items():
            text = re.sub(abbr, full, text)
            
        # Remove punctuation and extra whitespace
        text = re.sub(r'[^\\w\\s]', '', text)
        text = ' '.join(text.split())
        
        # Extract number and main part of address
        match = re.match(r'^(\\d+)\\s+(.*)', text)
        if match:
            number = match.group(1)
            rest = match.group(2)
            # Take first few words of street name to avoid apartment numbers etc.
            street_name = ' '.join(rest.split()[:2])
            return f"{number} {street_name}"
            
        return text # fallback to just cleaned text

    def _clean_dob(self, text: str) -> str:
        """Cleans date of birth to a consistent format of digits only."""
        if not text:
            return ""
        return re.sub(r'\\D', '', text)

    def _normalize_name(self, name: str, is_entered_field: bool = False) -> str:
        """Normalizes names to handle format differences and medical titles."""
        if not name:
            return ""
        
        clean_name = ' '.join(name.strip().split())
        
        medical_titles = ['md', 'm.d.', 'do', 'd.o.', 'np', 'n.p.', 'pa', 'p.a.', 
                         'pharmd', 'pharm.d.', 'rph', 'r.ph.', 'dds', 'd.d.s.', 
                         'dmd', 'd.m.d.', 'dvm', 'd.v.m.', 'phd', 'ph.d.', 
                         'rn', 'r.n.', 'lpn', 'l.p.n.', 'dr', 'dr.', 'ma', 'm.a.']
        
        for title in medical_titles:
            pattern = r'\b' + re.escape(title).replace(r'\.', r'\.?') + r'\b'
            clean_name = re.sub(pattern, '', clean_name, flags=re.IGNORECASE)
        
        clean_name = re.sub(r'\s+(jr|sr|iii|ii|iv)$', '', clean_name, flags=re.IGNORECASE)
        clean_name = clean_name.replace('|', ' ').replace('_', ' ')
        clean_name = ' '.join(clean_name.split())
        
        clean_name = self._standardize_name_format(clean_name)
        
        if '/' in clean_name:
            return " / ".join([self._standardize_name_format(p.strip()).lower() for p in clean_name.split('/') if p.strip()])
        
        return clean_name.lower()
    
    def _standardize_name_format(self, name: str) -> str:
        """Convert name to standardized 'First Middle Last' format."""
        if not name:
            return ""
        
        name = name.strip()
        
        if ',' in name:
            parts = [p.strip() for p in name.split(',')]
            if len(parts) == 2 and parts[0] and parts[1]:
                last_name = parts[0]
                first_middle = parts[1]
                return f"{first_middle} {last_name}"
        
        return name

    def verify_fields(self, ocr_results: Dict[str, Tuple[str, str]]) -> Dict[str, Any]:
        """Verifies all configured fields against their source locations."""
        results = {}
        fields_config = self.config["regions"]["fields"]
        thresholds = self.config["thresholds"]

        # Process all fields with fuzzy matching only (small area OCR+AI removed)
        for field_name, (entered_text, source_text) in ocr_results.items():
            if field_name not in fields_config:
                continue

            field_config = fields_config[field_name]
            threshold = thresholds[field_config["threshold_key"]]
            
            score, is_match = 0.0, False
            cleaned_entered, cleaned_source = "", ""

            # Check for empty fields first - fail immediately without processing
            if not entered_text.strip() or not source_text.strip():
                score = 0.0
                is_match = False
                cleaned_entered = entered_text if entered_text.strip() else "[EMPTY]"
                cleaned_source = source_text if source_text.strip() else "[EMPTY]"
                
                empty_field_type = []
                if not entered_text.strip():
                    empty_field_type.append("entered")
                if not source_text.strip():
                    empty_field_type.append("source")
                    
                logging.warning(f"Field {field_name} has empty {'/'.join(empty_field_type)} text - setting score to 0")
                
            else:
                # Use fuzzy matching for all small area fields
                if field_name in ["patient_name", "prescriber_name"]:
                    cleaned_entered = self._normalize_name(entered_text, is_entered_field=True)
                    cleaned_source = self._normalize_name(source_text, is_entered_field=False)
                elif field_name == "drug_name":
                    cleaned_entered = self._clean_drug_name(entered_text)
                    cleaned_source = self._clean_drug_name(source_text)
                elif field_name in ["direction_sig", "sig", "directions"]:
                    cleaned_entered = self._clean_sig_text(entered_text)
                    cleaned_source = self._clean_sig_text(source_text)
                elif field_name == "patient_dob":
                    cleaned_entered = self._clean_dob(entered_text)
                    cleaned_source = self._clean_dob(source_text)
                elif field_name == "patient_phone":
                    cleaned_entered = self._clean_phone_number(entered_text)
                    cleaned_source = self._clean_phone_number(source_text)
                elif field_name in ["patient_address", "prescriber_address"]:
                    cleaned_entered = self._normalize_address(entered_text)
                    cleaned_source = self._normalize_address(source_text)
                else:
                    cleaned_entered = self._clean_text(entered_text)
                    cleaned_source = self._clean_text(source_text)
                
                # Calculate score based on field type using existing logic
                if cleaned_entered and cleaned_source:
                    if field_name == "drug_name":
                        score, is_match = self._enhanced_drug_name_match(cleaned_entered, cleaned_source, threshold)
                    elif field_name == "prescriber_name":
                        score, is_match = self._match_prescriber_names(entered_text, source_text, threshold)
                    elif field_name == "patient_name":
                        score, is_match = self._enhanced_name_match(entered_text, source_text, threshold)
                    else:
                        score = fuzz.ratio(cleaned_entered, cleaned_source)
                        is_match = score >= threshold
                else:
                    score = 0.0
                    is_match = False
            
            # Use debug logging for detailed field analysis
            log_field_details(field_name, entered_text, source_text, 
                            cleaned_entered, cleaned_source, score, threshold, is_match)
            
            results[field_name] = {
                "match": is_match, 
                "score": score, 
                "coords": tuple(field_config["entered"]),
                "entered_raw": entered_text,
                "source_raw": source_text,
                "entered_clean": cleaned_entered,
                "source_clean": cleaned_source,
                "threshold": threshold
            }
            
        return results
