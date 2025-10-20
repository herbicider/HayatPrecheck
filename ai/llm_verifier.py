#!/usr/bin/env python3
"""
LLM AI Verification Module
==========================

This module implements LLM-based verification using 2 large OCR regions
(similar to VLM regions) + EasyOCR text extraction + LLM comparison.

This is the middle ground between traditional OCR+fuzzy matching and
full VLM image analysis.
"""

import json
import logging
import time
import asyncio
import warnings
from typing import Dict, Tuple, Any, Optional
from PIL import Image
import pyautogui

# Suppress PyTorch DataLoader pin_memory warning when no GPU is available
warnings.filterwarnings("ignore", message=".*pin_memory.*no accelerator is found.*", category=UserWarning)

from core.ocr_provider import get_cached_ocr_provider
from ai.cpu_verifier import AI_Verifier


class LLM_Verifier:
    """
    LLM-based verification using OCR + LLM comparison
    
    This method:
    1. Uses 2 large regions (like VLM) instead of many small regions
    2. Extracts text using EasyOCR (configurable)
    3. Sends extracted data to LLM for comparison
    4. Returns structured verification results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.verification_config = config.get("verification_methods", {}).get("llm_ai", {})
        
        # Always use the global OCR provider setting for consistency
        self.ocr_provider_type = config.get("ocr_provider", "auto")
            
        self.advanced_settings = config.get("advanced_settings", {})
        
        # Initialize OCR provider
        self.ocr_provider = get_cached_ocr_provider(
            self.ocr_provider_type, 
            self.advanced_settings
        )
        
        # Initialize AI verifier
        self.ai_verifier = None
        self._init_ai_verifier()
        
        # Load field extraction patterns
        self.field_patterns = self._load_field_patterns()
        
        logging.info(f"LLM Verifier initialized with OCR provider: {self.ocr_provider_type}")
    
    def _init_ai_verifier(self):
        """Initialize the AI verifier with LLM configuration"""
        try:
            # Load LLM config from llm_config.json
            llm_config_file = "config/llm_config.json"
            import os
            if os.path.exists(llm_config_file):
                with open(llm_config_file, 'r') as f:
                    llm_config = json.load(f)
                self.ai_verifier = AI_Verifier(llm_config)
                logging.info("LLM: AI Verifier initialized successfully")
            else:
                logging.error(f"LLM: Configuration file {llm_config_file} not found")
        except Exception as e:
            logging.error(f"LLM: Failed to initialize AI verifier: {e}")
    
    def _load_field_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load field extraction patterns from llm_config.json"""
        try:
            llm_config_file = "config/llm_config.json"
            import os
            if os.path.exists(llm_config_file):
                with open(llm_config_file, 'r') as f:
                    llm_config = json.load(f)
                    
                patterns = llm_config.get("field_extraction_patterns", {})
                logging.info(f"LLM: Loaded field patterns for {len(patterns)} fields")
                return patterns
            else:
                logging.warning("LLM: llm_config.json not found, using default patterns")
                return self._get_default_patterns()
                
        except Exception as e:
            logging.error(f"LLM: Error loading field patterns: {e}")
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Get default field extraction patterns as fallback"""
        return {
            "patient_name": {
                "primary_patterns": ["Patient:", "patient:", "Patient Name:", "Name:"],
                "fallback_patterns": ["pt:", "PT:", "Pt Name:"]
            },
            "prescriber_name": {
                "primary_patterns": ["Written By:", "written by:", "Prescriber:", "Doctor:"],
                "fallback_patterns": ["Dr.", "Physician:", "MD:", "Supervisor:"]
            },
            "drug_name": {
                "primary_patterns": ["Item:", "item:", "Medication:", "Drug:"],
                "fallback_patterns": ["Med:", "Rx:", "Medicine:"]
            },
            "direction_sig": {
                "primary_patterns": ["Directions:", "directions:", "Sig:", "Instructions:"],
                "fallback_patterns": ["Take:", "Use:", "Apply:"]
            }
        }
    
    def verify_with_llm_ai(self) -> Dict[str, Dict[str, Any]]:
        """
        Perform verification using LLM AI method with parallel processing
        
        Returns:
            Dictionary with verification results for each field
        """
        if not self.ai_verifier:
            logging.error("LLM: AI verifier not initialized")
            return {}
        
        try:
            total_start_time = time.time()
            
            # Take screenshot
            screenshot_start = time.time()
            screenshot = pyautogui.screenshot()
            screenshot_time = time.time() - screenshot_start
            
            # Extract text from 2 regions using OCR with parallel field parsing
            ocr_start = time.time()
            ocr_results = self._extract_ocr_from_regions(screenshot)
            ocr_time = time.time() - ocr_start
            
            if not ocr_results:
                logging.error("LLM: No OCR data extracted")
                return {}
            
            # Send to LLM for comparison
            llm_start = time.time()
            llm_scores = self.ai_verifier.verify_with_ai("batch_verification", ocr_results)
            llm_time = time.time() - llm_start
            
            logging.info(f"LLM: Raw scores returned: {llm_scores}")
            logging.info(f"LLM: OCR results keys: {list(ocr_results.keys())}")
            
            # Convert to verification results format
            convert_start = time.time()
            results = self._convert_to_verification_results(llm_scores, ocr_results)
            convert_time = time.time() - convert_start
            
            total_time = time.time() - total_start_time
            
            # Performance summary
            logging.info(f"LLM Performance: Screenshot={screenshot_time:.3f}s, OCR+Parse={ocr_time:.3f}s, "
                        f"LLM={llm_time:.3f}s, Convert={convert_time:.3f}s, Total={total_time:.3f}s")
            
            logging.info(f"LLM: Verification completed for {len(results)} fields")
            return results
            
        except Exception as e:
            logging.error(f"LLM: Error during verification: {e}")
            return {}
    
    def _extract_ocr_from_regions(self, screenshot: Image.Image) -> Dict[str, Tuple[str, str]]:
        """
        Extract OCR text from 2 large regions and parse into field data
        
        Args:
            screenshot: PIL Image of current screen
            
        Returns:
            Dictionary mapping field names to (entered_text, source_text) tuples
        """
        try:
            # Get region coordinates
            llm_regions = self.config.get("llm_regions", {})
            if not llm_regions:
                # Fallback to VLM regions if llm_regions not configured
                vlm_config_file = "config/vlm_config.json"
                import os
                if os.path.exists(vlm_config_file):
                    with open(vlm_config_file, 'r') as f:
                        vlm_config = json.load(f)
                    llm_regions = vlm_config.get("vlm_regions", {})
            
            data_entry_region = llm_regions.get("data_entry", [0, 0, 100, 100])
            source_region = llm_regions.get("source", [0, 0, 100, 100])
            
            logging.debug(f"LLM: Using regions - data_entry: {data_entry_region}, source: {source_region}")
            
            # Extract text from both regions
            data_entry_text = self.ocr_provider.get_text_from_region(
                screenshot, tuple(data_entry_region), "llm_data_entry"
            )
            source_text = self.ocr_provider.get_text_from_region(
                screenshot, tuple(source_region), "llm_source"
            )
            
            logging.info(f"LLM OCR: Data entry text: '{data_entry_text}'")
            logging.info(f"LLM OCR: Source text: '{source_text}'")
            
            # Parse OCR text into field data
            ocr_results = self._parse_ocr_into_fields(data_entry_text, source_text)
            
            return ocr_results
            
        except Exception as e:
            logging.error(f"LLM: Error extracting OCR from regions: {e}")
            return {}
    
    def _parse_ocr_into_fields(self, data_entry_text: str, source_text: str) -> Dict[str, Tuple[str, str]]:
        """
        Parse OCR text into field data - simplified approach that leverages LLM intelligence
        
        Strategy:
        - Left side (entered): Simple extraction since UI is consistent  
        - Right side (source): Send full OCR text - let LLM parse it
        - Let GPT-OSS-20B do the heavy lifting instead of rigid pattern matching
        
        Args:
            data_entry_text: OCR text from data entry region (consistent UI)
            source_text: OCR text from source region (variable formats)
            
        Returns:
            Dictionary mapping field names to (entered_text, source_text) tuples
        """
        try:
            start_time = time.time()
            
            # Extract entered fields using simple patterns (UI is consistent)
            entered_fields = self._extract_entered_fields_simple(data_entry_text)
            
            # For source: send everything to LLM, don't try to be smart
            # The LLM gets the full context and can handle different formats
            source_full_text = source_text.strip()
            
            # Create field mapping with entered data + full source text for each field
            ocr_results = {}
            field_names = ["patient_name", "prescriber_name", "drug_name", "direction_sig"]
            
            for field_name in field_names:
                entered = entered_fields.get(field_name, "")
                # Send full source text for each field - let LLM extract what it needs
                ocr_results[field_name] = (entered, source_full_text)
                
                logging.debug(f"LLM Simple: {field_name} - entered: '{entered}' | source: [FULL_TEXT]")
            
            elapsed_time = time.time() - start_time
            logging.info(f"LLM: Simplified OCR parsing completed in {elapsed_time:.3f}s")
            logging.info(f"LLM: Entered fields extracted: {[(k, v) for k, v in entered_fields.items() if v]}")
            logging.info(f"LLM: Source text length: {len(source_full_text)} chars")
            
            return ocr_results
            
        except Exception as e:
            logging.error(f"LLM: Error in simplified OCR parsing: {e}")
            return self._create_fallback_results(data_entry_text, source_text)
    
    def _extract_entered_fields_simple(self, data_entry_text: str) -> Dict[str, str]:
        """
        Simple extraction for entered data - UI is consistent, so basic patterns work
        
        Args:
            data_entry_text: OCR text from data entry region
            
        Returns:
            Dictionary mapping field names to extracted text
        """
        import re
        
        fields = {
            "patient_name": "",
            "prescriber_name": "",
            "drug_name": "",
            "direction_sig": ""
        }
        
        try:
            # Use regex patterns to extract fields directly from the OCR text
            full_text = data_entry_text
            
            # 1. Patient Name - look for "Patient: [name]" pattern
            patient_match = re.search(r'Patient:\s*([A-Za-z,\s]+?)(?:\s+Address:|$)', full_text, re.IGNORECASE)
            if patient_match:
                patient_name = patient_match.group(1).strip()
                # Clean up common OCR artifacts and normalize name format
                patient_name = re.sub(r'[,\s]+$', '', patient_name)  # Remove trailing commas/spaces
                # Convert "Gonzalez,NoRMA" to "NoRMA Gonzalez" (First Last format)
                if ',' in patient_name:
                    parts = [p.strip() for p in patient_name.split(',')]
                    if len(parts) == 2:
                        patient_name = f"{parts[1]} {parts[0]}"
                fields["patient_name"] = patient_name
            
            # 2. Prescriber Name - look for "Written By. [name]" pattern  
            prescriber_match = re.search(r'Written By\.?\s*([A-Za-z\.\s]+?)\s*(?:M\.?D\.?|Address:|$)', full_text, re.IGNORECASE)
            if prescriber_match:
                prescriber_name = prescriber_match.group(1).strip()
                # Clean up dots and extra spaces
                prescriber_name = re.sub(r'\.+', ' ', prescriber_name)  # Replace dots with spaces
                prescriber_name = re.sub(r'\s+', ' ', prescriber_name)  # Normalize spaces
                prescriber_name = prescriber_name.strip()
                fields["prescriber_name"] = prescriber_name
            
            # 3. Drug Name - look for "Item: [drug]" pattern
            drug_match = re.search(r'Item:\s*([A-Za-z0-9\s%]+?)(?:\s+Quantity|$)', full_text, re.IGNORECASE)
            if drug_match:
                drug_name = drug_match.group(1).strip()
                fields["drug_name"] = drug_name
            
            # 4. Directions - look for directions pattern after "Directions" or "Sig"
            # Handle the complex format: "Directions (Sig Codes or Text or [ Literal Text ] ): (pr 4/2) [actual directions]"
            directions_match = re.search(r'Directions[^:]*:\s*(?:\([^)]*\)\s*)?(.+?)(?:Spanish:|$)', full_text, re.IGNORECASE | re.DOTALL)
            if directions_match:
                directions = directions_match.group(1).strip()
                # Clean up common OCR artifacts in directions
                directions = re.sub(r'\s+', ' ', directions)  # Normalize spaces
                fields["direction_sig"] = directions
            
            logging.info(f"LLM Simple: Extracted entered fields: {[(k, v) for k, v in fields.items() if v]}")
            return fields
            
        except Exception as e:
            logging.error(f"LLM Simple: Error extracting entered fields: {e}")
            return {"patient_name": "", "prescriber_name": "", "drug_name": "", "direction_sig": ""}
    
    def _create_fallback_results(self, data_entry_text: str, source_text: str) -> Dict[str, Tuple[str, str]]:
        """
        Create minimal fallback results when extraction fails
        
        Args:
            data_entry_text: OCR text from data entry region
            source_text: OCR text from source region
            
        Returns:
            Basic field mapping with full text for LLM to handle
        """
        # If extraction fails completely, still send the full text to LLM
        field_names = ["patient_name", "prescriber_name", "drug_name", "direction_sig"]
        return {field_name: (data_entry_text, source_text) for field_name in field_names}
    
    # Field extraction is now simplified - let the LLM handle the complexity!
    
    def _convert_to_verification_results(
        self, 
        llm_scores: Dict[str, int], 
        ocr_results: Dict[str, Tuple[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Convert LLM scores and OCR results to verification results format
        
        Args:
            llm_scores: Dictionary of field names to LLM confidence scores
            ocr_results: Dictionary of field names to (entered, source) text tuples
            
        Returns:
            Dictionary in verification results format
        """
        results = {}
        thresholds = self.config.get("thresholds", {})
        
        # Map field names to threshold keys
        field_threshold_map = {
            "patient_name": "patient",
            "prescriber_name": "prescriber",
            "drug_name": "drug",
            "direction_sig": "sig"
        }
        
        logging.info(f"LLM Convert: Processing {len(ocr_results)} fields")
        logging.info(f"LLM Convert: LLM scores received: {llm_scores}")
        logging.info(f"LLM Convert: LLM score types: {[(k, type(v)) for k, v in llm_scores.items()]}")
        logging.info(f"LLM Convert: OCR results fields: {list(ocr_results.keys())}")
        
        for field_name in ocr_results.keys():
            entered_text, source_text = ocr_results[field_name]
            score = llm_scores.get(field_name, 0)
            
            # DETAILED DEBUG: Show exact field mapping
            logging.info(f"LLM Convert: Looking for '{field_name}' in LLM scores")
            logging.info(f"LLM Convert: Available LLM keys: {list(llm_scores.keys())}")
            logging.info(f"LLM Convert: Raw score lookup result: {repr(score)} (type: {type(score)})")
            
            # Convert score to float if it's not already
            try:
                score = float(score)
            except (ValueError, TypeError):
                logging.error(f"LLM Convert: Invalid score type for {field_name}: {score} ({type(score)})")
                score = 0.0
            
            logging.info(f"LLM Convert: Final score for '{field_name}': {score}")
            
            # Get threshold
            threshold_key = field_threshold_map.get(field_name, field_name)
            threshold = thresholds.get(threshold_key, 70)
            
            # Determine match
            match = score >= threshold
            
            # Get coordinates for overlay (use existing field regions if available)
            field_coords = self.config.get("regions", {}).get("fields", {}).get(field_name, {}).get("entered", [0, 0, 0, 0])
            
            results[field_name] = {
                "entered": entered_text,
                "source": source_text,
                "score": score,
                "match": match,
                "threshold": threshold,
                "method": "llm_ai",
                "coords": field_coords
            }
            
            logging.info(f"LLM Convert: Created result for '{field_name}': score={score}, match={match}, threshold={threshold}")
        
        logging.info(f"LLM Convert: Final results summary: {[(k, v['score']) for k, v in results.items()]}")
        return results