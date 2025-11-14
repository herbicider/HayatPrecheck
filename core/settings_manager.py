#!/usr/bin/env python3
"""
Unified Settings Management for Pharmacy Verification System
============================================================

This module combines the functionality of both the CLI and GUI settings tools
into a unified interface that can be used by the Streamlit application or
standalone as needed.

Features:
- Configuration loading and saving
- Coordinate validation and adjustment
- Threshold management
- Backup and restore functionality
- OCR testing on regions

Usage:
    from settings_manager import SettingsManager
    
    settings = SettingsManager()
    settings.load_config()
    settings.validate_coordinates()
"""

import json
import os
import time
import shutil
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import pyautogui
from PIL import Image, ImageEnhance
import pytesseract
import logging
import re
from dotenv import load_dotenv

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

class SettingsManager:
    """Unified settings management for the pharmacy verification system"""
    
    def __init__(self, config_file: str = "config/config.json", vlm_config_file: Optional[str] = None, llm_config_file: Optional[str] = None):
        self.config_file = config_file
        self.vlm_config_file = vlm_config_file or os.path.join("config", "vlm_config.json")
        self.llm_config_file = llm_config_file or os.path.join("config", "llm_config.json")
        self.config: Optional[Dict[str, Any]] = None
        self.vlm_config: Optional[Dict[str, Any]] = None
        self.llm_config: Optional[Dict[str, Any]] = None
        self.backup_dir = "config_backups"
        
        # Ensure backup directory exists
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
    
    def load_config(self) -> bool:
        """Load configuration from config.json file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                self.logger.debug(f"Configuration loaded from {self.config_file}")  # Changed from INFO to DEBUG
                return True
            else:
                self.logger.error(f"Configuration file {self.config_file} not found")
                return False
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def load_vlm_config(self) -> bool:
        """Load VLM configuration from vlm_config.json file with environment variable substitution"""
        try:
            if os.path.exists(self.vlm_config_file):
                with open(self.vlm_config_file, 'r', encoding='utf-8') as f:
                    raw_config = json.load(f)
                
                # Substitute environment variables
                self.vlm_config = substitute_env_vars(raw_config)
                self.logger.debug(f"VLM configuration loaded from {self.vlm_config_file} with environment variable substitution")
                return True
            else:
                self.logger.error(f"VLM configuration file {self.vlm_config_file} not found")
                return False
        except Exception as e:
            self.logger.error(f"Error loading VLM configuration: {e}")
            return False

    def load_llm_config(self) -> bool:
        """Load LLM configuration from llm_config.json file"""
        try:
            if os.path.exists(self.llm_config_file):
                with open(self.llm_config_file, 'r', encoding='utf-8') as f:
                    self.llm_config = json.load(f)
                self.logger.debug(f"LLM configuration loaded from {self.llm_config_file}")
                return True
            else:
                self.logger.error(f"LLM configuration file {self.llm_config_file} not found")
                return False
        except Exception as e:
            self.logger.error(f"Error loading LLM configuration: {e}")
            return False
    
    def save_config(self, create_backup: bool = False) -> bool:
        """Save configuration to config.json file"""
        if not self.config:
            self.logger.error("No configuration to save")
            return False
        
        try:
            # Create backup only if requested
            if create_backup:
                self.create_backup()
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration saved to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
            return False
    
    def save_vlm_config(self, create_backup: bool = False) -> bool:
        """Save VLM configuration to vlm_config.json file"""
        if not self.vlm_config:
            self.logger.error("No VLM configuration to save")
            return False
        
        try:
            # Create backup only if requested
            if create_backup:
                self.create_vlm_backup()
            
            with open(self.vlm_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.vlm_config, f, indent=2)
            self.logger.info(f"VLM configuration saved to {self.vlm_config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving VLM configuration: {e}")
            return False

    def save_llm_config(self, create_backup: bool = False) -> bool:
        """Save LLM configuration to llm_config.json file"""
        if not self.llm_config:
            self.logger.error("No LLM configuration to save")
            return False
        
        try:
            # Create backup only if requested
            if create_backup:
                self.create_llm_backup()
            
            with open(self.llm_config_file, 'w', encoding='utf-8') as f:
                json.dump(self.llm_config, f, indent=2)
            self.logger.info(f"LLM configuration saved to {self.llm_config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving LLM configuration: {e}")
            return False
    
    def create_backup(self) -> str:
        """Create a backup of the current configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"config_backup_{timestamp}.json")
        
        try:
            if os.path.exists(self.config_file):
                shutil.copy2(self.config_file, backup_file)
                self.logger.info(f"Backup created: {backup_file}")
                
                # Clean up old backups (keep last 10)
                self.cleanup_old_backups()
                
                return backup_file
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
        
        return ""
    
    def create_vlm_backup(self) -> str:
        """Create a backup of the current VLM configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"vlm_config_backup_{timestamp}.json")
        
        try:
            if os.path.exists(self.vlm_config_file):
                shutil.copy2(self.vlm_config_file, backup_file)
                self.logger.info(f"VLM backup created: {backup_file}")
                
                # Clean up old backups (keep last 10)
                self.cleanup_old_vlm_backups()
                
                return backup_file
        except Exception as e:
            self.logger.error(f"Error creating VLM backup: {e}")
        
        return ""

    def create_llm_backup(self) -> str:
        """Create a backup of the current LLM configuration"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = os.path.join(self.backup_dir, f"llm_config_backup_{timestamp}.json")
        
        try:
            if os.path.exists(self.llm_config_file):
                shutil.copy2(self.llm_config_file, backup_file)
                self.logger.info(f"LLM backup created: {backup_file}")
                
                # Clean up old backups (keep last 10)
                self.cleanup_old_llm_backups()
                
                return backup_file
        except Exception as e:
            self.logger.error(f"Error creating LLM backup: {e}")
        
        return ""
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Remove old backup files, keeping only the most recent ones"""
        try:
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("config_backup_") and filename.endswith(".json"):
                    filepath = os.path.join(self.backup_dir, filename)
                    backup_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove excess backups
            for filepath, _ in backup_files[keep_count:]:
                os.remove(filepath)
                self.logger.info(f"Removed old backup: {filepath}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up backups: {e}")
    
    def cleanup_old_vlm_backups(self, keep_count: int = 10):
        """Remove old VLM backup files, keeping only the most recent ones"""
        try:
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("vlm_config_backup_") and filename.endswith(".json"):
                    filepath = os.path.join(self.backup_dir, filename)
                    backup_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove excess backups
            for filepath, _ in backup_files[keep_count:]:
                os.remove(filepath)
                self.logger.info(f"Removed old VLM backup: {filepath}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up VLM backups: {e}")

    def cleanup_old_llm_backups(self, keep_count: int = 10):
        """Remove old LLM backup files, keeping only the most recent ones"""
        try:
            backup_files = []
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("llm_config_backup_") and filename.endswith(".json"):
                    filepath = os.path.join(self.backup_dir, filename)
                    backup_files.append((filepath, os.path.getmtime(filepath)))
            
            # Sort by modification time (newest first)
            backup_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove excess backups
            for filepath, _ in backup_files[keep_count:]:
                os.remove(filepath)
                self.logger.info(f"Removed old LLM backup: {filepath}")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up LLM backups: {e}")
    
    def restore_backup(self, backup_file: str) -> bool:
        """Restore configuration from a backup file"""
        try:
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, self.config_file)
                self.load_config()  # Reload the restored config
                self.logger.info(f"Configuration restored from {backup_file}")
                return True
            else:
                self.logger.error(f"Backup file not found: {backup_file}")
                return False
        except Exception as e:
            self.logger.error(f"Error restoring backup: {e}")
            return False
    
    def get_available_backups(self) -> List[str]:
        """Get list of available backup files"""
        backups = []
        try:
            for filename in os.listdir(self.backup_dir):
                if filename.startswith("config_backup_") and filename.endswith(".json"):
                    backups.append(os.path.join(self.backup_dir, filename))
            backups.sort(reverse=True)  # Newest first
        except Exception as e:
            self.logger.error(f"Error listing backups: {e}")
        
        return backups
    
    def validate_coordinates(self) -> Dict[str, Any]:
        """Validate all coordinates in the configuration"""
        if not self.config:
            return {"valid": False, "error": "No configuration loaded"}
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "regions_checked": 0
        }
        
        try:
            # Get screen size
            screen_width, screen_height = pyautogui.size()
            
            # Check trigger region
            if "trigger" in self.config.get("regions", {}):
                trigger = self.config["regions"]["trigger"]
                result = self._validate_region("trigger", trigger, screen_width, screen_height)
                if not result["valid"]:
                    validation_results["errors"].extend(result["errors"])
                    validation_results["valid"] = False
                validation_results["regions_checked"] += 1
            
            # Check field regions
            fields = self.config.get("regions", {}).get("fields", {})
            for field_name, field_config in fields.items():
                for region_type in ["entered", "source"]:
                    if region_type in field_config:
                        region = field_config[region_type]
                        result = self._validate_region(
                            f"{field_name}.{region_type}", 
                            region, 
                            screen_width, 
                            screen_height
                        )
                        if not result["valid"]:
                            validation_results["errors"].extend(result["errors"])
                            validation_results["valid"] = False
                        validation_results["regions_checked"] += 1
            
        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {e}")
        
        return validation_results
    
    def _validate_region(self, name: str, coords: List[int], screen_width: int, screen_height: int) -> Dict[str, Any]:
        """Validate a single coordinate region"""
        result = {"valid": True, "errors": []}
        
        if len(coords) != 4:
            result["valid"] = False
            result["errors"].append(f"{name}: Invalid coordinate format (expected 4 values)")
            return result
        
        x1, y1, x2, y2 = coords
        
        # Check if coordinates are within screen bounds
        if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
            result["valid"] = False
            result["errors"].append(f"{name}: Negative coordinates not allowed")
        
        if x1 >= screen_width or x2 >= screen_width:
            result["valid"] = False
            result["errors"].append(f"{name}: X coordinates exceed screen width ({screen_width})")
        
        if y1 >= screen_height or y2 >= screen_height:
            result["valid"] = False
            result["errors"].append(f"{name}: Y coordinates exceed screen height ({screen_height})")
        
        # Check if region has valid dimensions
        if x2 <= x1 or y2 <= y1:
            result["valid"] = False
            result["errors"].append(f"{name}: Invalid region dimensions (x2 <= x1 or y2 <= y1)")
        
        return result
    
    def test_ocr_region(self, coords: List[int]) -> Dict[str, Any]:
        """Test OCR on a specific coordinate region"""
        result = {
            "success": False,
            "text": "",
            "error": "",
            "processing_time": 0
        }
        
        try:
            start_time = time.time()
            
            # Capture screenshot of the region
            x1, y1, x2, y2 = coords
            screenshot = pyautogui.screenshot(region=(x1, y1, x2 - x1, y2 - y1))
            
            # Enhance image for better OCR
            screenshot = screenshot.convert('L')  # Convert to grayscale
            enhancer = ImageEnhance.Contrast(screenshot)
            screenshot = enhancer.enhance(2.0)  # Increase contrast
            
            # Perform OCR
            text = pytesseract.image_to_string(screenshot).strip()
            
            result["success"] = True
            result["text"] = text
            result["processing_time"] = time.time() - start_time
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def update_threshold(self, field_type: str, value: int) -> bool:
        """Update a threshold value"""
        if not self.config:
            return False
        
        try:
            if "thresholds" not in self.config:
                self.config["thresholds"] = {}
            
            self.config["thresholds"][field_type] = value
            return True
        except Exception as e:
            self.logger.error(f"Error updating threshold: {e}")
            return False
    
    def update_timing(self, timing_key: str, value: float) -> bool:
        """Update a timing value"""
        if not self.config:
            return False
        
        try:
            if "timing" not in self.config:
                self.config["timing"] = {}
            
            self.config["timing"][timing_key] = value
            return True
        except Exception as e:
            self.logger.error(f"Error updating timing: {e}")
            return False
    
    def update_coordinates(self, region_path: str, coords: List[int]) -> bool:
        """Update coordinates for a specific region
        
        Args:
            region_path: Path like "trigger" or "fields.patient_name.entered"
            coords: List of 4 integers [x1, y1, x2, y2]
        """
        if not self.config:
            return False
        
        try:
            if "regions" not in self.config:
                self.config["regions"] = {}
            
            parts = region_path.split('.')
            
            if parts[0] == "trigger":
                self.config["regions"]["trigger"] = coords
            elif parts[0] == "fields" and len(parts) == 3:
                field_name, region_type = parts[1], parts[2]
                if "fields" not in self.config["regions"]:
                    self.config["regions"]["fields"] = {}
                if field_name not in self.config["regions"]["fields"]:
                    self.config["regions"]["fields"][field_name] = {}
                
                self.config["regions"]["fields"][field_name][region_type] = coords
            else:
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error updating coordinates: {e}")
            return False
    
    def export_config(self, filepath: str) -> bool:
        """Export configuration to a file"""
        if not self.config:
            return False
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Configuration exported to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error exporting configuration: {e}")
            return False
    
    def import_config(self, filepath: str) -> bool:
        """Import configuration from a file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Validate the imported configuration
            if self._validate_config_structure(imported_config):
                self.config = imported_config
                self.logger.info(f"Configuration imported from {filepath}")
                return True
            else:
                self.logger.error("Invalid configuration structure in imported file")
                return False
        except Exception as e:
            self.logger.error(f"Error importing configuration: {e}")
            return False
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validate that a configuration has the required structure"""
        required_sections = ["timing", "thresholds", "regions"]
        
        for section in required_sections:
            if section not in config:
                return False
        
        # Check regions structure
        regions = config.get("regions", {})
        if "trigger" not in regions or "fields" not in regions:
            return False
        
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get a default configuration template"""
        return {
            "timing": {
                "fast_polling_seconds": 0.2,
                "trigger_check_interval_seconds": 1.0,
                "same_prescription_wait_seconds": 3.0,
                "max_static_sleep_seconds": 2.0,
                "verification_wait_seconds": 0.5
            },
            "thresholds": {
                "patient": 70,
                "prescriber": 70,
                "drug": 70,
                "sig": 65
            },
            "regions": {
                "trigger": [5, 56, 144, 79],
                "fields": {
                    "patient_name": {
                        "entered": [91, 147, 270, 165],
                        "source": [518, 400, 840, 419],
                        "score_fn": "token_sort_ratio",
                        "threshold_key": "patient"
                    },
                    "prescriber_name": {
                        "entered": [91, 222, 290, 239],
                        "source": [516, 276, 961, 303],
                        "score_fn": "token_sort_ratio",
                        "threshold_key": "prescriber"
                    }
                }
            }
        }
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to default values"""
        try:
            self.config = self.get_default_config()
            self.logger.info("Configuration reset to defaults")
            return True
        except Exception as e:
            self.logger.error(f"Error resetting to defaults: {e}")
            return False

# Convenience functions for backward compatibility
def load_config(config_file: str = "config/config.json") -> Optional[Dict[str, Any]]:
    """Load configuration (standalone function)"""
    manager = SettingsManager(config_file)
    if manager.load_config():
        return manager.config
    return None

def save_config(config: Dict[str, Any], config_file: str = "config/config.json") -> bool:
    """Save configuration (standalone function)"""
    manager = SettingsManager(config_file)
    manager.config = config
    return manager.save_config()

if __name__ == "__main__":
    # Simple CLI interface for testing
    import sys
    
    manager = SettingsManager()
    
    if len(sys.argv) < 2:
        print("Usage: python settings_manager.py [command]")
        print("Commands: load, validate, backup, test-ocr")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "load":
        if manager.load_config():
            print("Configuration loaded successfully")
            print(json.dumps(manager.config, indent=2))
        else:
            print("Failed to load configuration")
    
    elif command == "validate":
        if manager.load_config():
            results = manager.validate_coordinates()
            print(f"Validation results: {results}")
        else:
            print("Failed to load configuration")
    
    elif command == "backup":
        backup_file = manager.create_backup()
        if backup_file:
            print(f"Backup created: {backup_file}")
        else:
            print("Failed to create backup")
    
    elif command == "test-ocr":
        if len(sys.argv) < 6:
            print("Usage: python settings_manager.py test-ocr x1 y1 x2 y2")
            sys.exit(1)
        
        coords = [int(x) for x in sys.argv[2:6]]
        result = manager.test_ocr_region(coords)
        print(f"OCR Test Results: {result}")
    
    else:
        print(f"Unknown command: {command}")
