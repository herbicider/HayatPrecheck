#!/usr/bin/env python3
"""
Command-line Settings Helper for Pharmacy Verification System
=============================================================

A simple command-line tool for viewing and editing coordinates and general 
settings when the GUI version is not available. This tool can help inspect 
and modify the config.json file coordinates and program settings.

Usage:
    python3 settings_cli.py [command] [options]
    
Commands:
    show       - Display current coordinates
    edit       - Edit coordinates for a specific field
    validate   - Validate all coordinates
    backup     - Create backup of config.json
    settings   - View or modify general settings

Example:
    python3 settings_cli.py show
    python3 settings_cli.py edit patient_name entered
    python3 settings_cli.py settings show
    python3 settings_cli.py settings set threshold patient 70
"""

import json
import os
import sys
import shutil
import time
from typing import Dict, Any, List, Optional

class SettingsCLI:
    def __init__(self):
        self.config: Optional[Dict[str, Any]] = None
        self.load_config()
    
    def load_config(self) -> bool:
        """Load configuration from config.json file."""
        try:
            config_path = "config.json"
            if not os.path.exists(config_path):
                config_path = os.path.join(os.path.dirname(__file__), "config.json")
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def save_config(self):
        """Save configuration to config.json file."""
        if not self.config:
            print("No configuration to save")
            return False
            
        try:
            config_path = "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print("Configuration saved successfully!")
            return True
            
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
    
    def show_coordinates(self):
        """Display all current coordinates."""
        if not self.config:
            print("No configuration loaded")
            return
        
        print("Current Coordinates:")
        print("===================")
        
        # Show trigger region
        if "trigger" in self.config["regions"]:
            coords = self.config["regions"]["trigger"]
            print(f"trigger: [{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]")
        
        # Show field regions
        print("\nFields:")
        for field_name, field_config in self.config["regions"]["fields"].items():
            print(f"\n{field_name}:")
            for region_type in ["entered", "source"]:
                if region_type in field_config:
                    coords = field_config[region_type]
                    print(f"  {region_type}: [{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]")
    
    def edit_coordinates(self, field_name: str, region_type: str):
        """Edit coordinates for a specific field and region."""
        if not self.config:
            print("No configuration loaded")
            return
        
        try:
            if field_name == "trigger":
                current_coords = self.config["regions"]["trigger"]
                coord_path = "trigger"
            else:
                current_coords = self.config["regions"]["fields"][field_name][region_type]
                coord_path = f"{field_name}.{region_type}"
            
            print(f"\nEditing coordinates for {coord_path}")
            print(f"Current: [{current_coords[0]}, {current_coords[1]}, {current_coords[2]}, {current_coords[3]}]")
            print("Enter new coordinates (x1, y1, x2, y2):")
            
            try:
                x1 = int(input("X1 (left): "))
                y1 = int(input("Y1 (top): "))
                x2 = int(input("X2 (right): "))
                y2 = int(input("Y2 (bottom): "))
                
                # Validate
                if x1 >= x2 or y1 >= y2:
                    print("Error: Invalid coordinates (x1 must be < x2, y1 must be < y2)")
                    return
                
                # Update config
                new_coords = [x1, y1, x2, y2]
                if field_name == "trigger":
                    self.config["regions"]["trigger"] = new_coords
                else:
                    self.config["regions"]["fields"][field_name][region_type] = new_coords
                
                print(f"Updated {coord_path} to: {new_coords}")
                
                # Ask to save
                save = input("Save changes? (y/n): ").lower().strip()
                if save == 'y':
                    self.save_config()
                
            except ValueError:
                print("Error: Please enter valid integer coordinates")
                
        except KeyError:
            print(f"Error: Field '{field_name}' or region '{region_type}' not found")
    
    def validate_coordinates(self):
        """Validate all coordinates for potential issues."""
        if not self.config:
            print("No configuration loaded")
            return
        
        issues = []
        
        # Check trigger region
        if "trigger" in self.config["regions"]:
            coords = self.config["regions"]["trigger"]
            if len(coords) != 4:
                issues.append("Trigger region: Invalid coordinate count")
            elif coords[0] >= coords[2] or coords[1] >= coords[3]:
                issues.append("Trigger region: Invalid coordinate order")
        
        # Check field regions
        for field_name, field_config in self.config["regions"]["fields"].items():
            for region_type in ["entered", "source"]:
                if region_type in field_config:
                    coords = field_config[region_type]
                    if len(coords) != 4:
                        issues.append(f"{field_name} {region_type}: Invalid coordinate count")
                    elif coords[0] >= coords[2] or coords[1] >= coords[3]:
                        issues.append(f"{field_name} {region_type}: Invalid coordinate order")
        
        # Show results
        if issues:
            print("Validation Issues Found:")
            print("=======================")
            for issue in issues:
                print(f"- {issue}")
        else:
            print("Validation Complete: All regions are valid!")
    
    def create_backup(self):
        """Create a backup of the current configuration."""
        config_path = "config.json"
        if not os.path.exists(config_path):
            print("No config.json file found")
            return
            
        backup_path = f"config_backup_{int(time.time())}.json"
        try:
            shutil.copy2(config_path, backup_path)
            print(f"Backup created: {backup_path}")
        except Exception as e:
            print(f"Error creating backup: {e}")
    
    def show_settings(self):
        """Display current general settings."""
        if not self.config:
            print("No configuration loaded")
            return
        
        print("Current Settings:")
        print("=================")
        
        # Show thresholds
        print("\nMatching Thresholds:")
        thresholds = self.config.get("thresholds", {})
        for field, threshold in thresholds.items():
            print(f"  {field}: {threshold}%")
        
        # Show automation settings
        print("\nAutomation Settings:")
        automation = self.config.get("automation", {})
        enabled = automation.get("send_key_on_all_match", False)
        key = automation.get("key_on_all_match", "f12")
        delay = automation.get("key_delay_seconds", 0.5)
        
        print(f"  Enabled: {enabled}")
        print(f"  Key to send: {key}")
        print(f"  Delay: {delay}s")
    
    def set_setting(self, setting_type: str, setting_key: str, setting_value: str):
        """Set a configuration setting."""
        if not self.config:
            print("No configuration loaded")
            return
        
        if setting_type == "threshold":
            try:
                value = int(setting_value)
                if 0 <= value <= 100:
                    if "thresholds" not in self.config:
                        self.config["thresholds"] = {}
                    self.config["thresholds"][setting_key] = value
                    print(f"Set {setting_key} threshold to {value}%")
                    self.save_config()
                else:
                    print("Threshold must be between 0 and 100")
            except ValueError:
                print("Threshold must be a number")
        
        elif setting_type == "automation":
            if "automation" not in self.config:
                self.config["automation"] = {}
            
            if setting_key == "enable":
                self.config["automation"]["send_key_on_all_match"] = True
                print("Automation enabled")
                self.save_config()
            elif setting_key == "disable":
                self.config["automation"]["send_key_on_all_match"] = False
                print("Automation disabled")
                self.save_config()
            elif setting_key == "key":
                self.config["automation"]["key_on_all_match"] = setting_value
                print(f"Automation key set to {setting_value}")
                self.save_config()
            elif setting_key == "delay":
                try:
                    delay = float(setting_value)
                    if delay >= 0:
                        self.config["automation"]["key_delay_seconds"] = delay
                        print(f"Automation delay set to {delay}s")
                        self.save_config()
                    else:
                        print("Delay must be non-negative")
                except ValueError:
                    print("Delay must be a number")
            else:
                print(f"Unknown automation setting: {setting_key}")
        
        else:
            print(f"Unknown setting type: {setting_type}")

def main():
    """Main command-line interface."""
    helper = SettingsCLI()
    
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    command = sys.argv[1].lower()
    
    if command == "show":
        helper.show_coordinates()
    
    elif command == "edit":
        if len(sys.argv) < 4:
            print("Usage: python3 coordinate_helper.py edit <field_name> <region_type>")
            print("Example: python3 coordinate_helper.py edit patient_name entered")
            return
        field_name = sys.argv[2]
        region_type = sys.argv[3]
        helper.edit_coordinates(field_name, region_type)
    
    elif command == "validate":
        helper.validate_coordinates()
    
    elif command == "backup":
        helper.create_backup()
    
    elif command == "settings":
        if len(sys.argv) < 3:
            print("Usage: python3 coordinate_helper.py settings <show|set> [options]")
            print("Examples:")
            print("  python3 coordinate_helper.py settings show")
            print("  python3 coordinate_helper.py settings set threshold patient 70")
            print("  python3 coordinate_helper.py settings set automation enable")
            print("  python3 coordinate_helper.py settings set automation key f11")
            print("  python3 coordinate_helper.py settings set automation delay 1.0")
            return
        
        action = sys.argv[2]
        if action == "show":
            helper.show_settings()
        elif action == "set" and len(sys.argv) >= 6:
            setting_type = sys.argv[3]
            setting_key = sys.argv[4]
            setting_value = sys.argv[5]
            helper.set_setting(setting_type, setting_key, setting_value)
        else:
            print("Invalid settings command")
    
    else:
        print(f"Unknown command: {command}")
        print(__doc__)

if __name__ == "__main__":
    main()
