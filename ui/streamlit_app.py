#!/usr/bin/env python3
"""
Simple Streamlit Pharmacy Verification System
============================================

A simplified web-based interface for configuring pharmacy verification coordinates.
Much easier to use than the complex version.

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import json
import os
import sys
import time
import logging
import re
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger_config import setup_logging
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import threading
import subprocess
import sys
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from ui.streamlit_vlm_page import vlm_settings_page

class SimplePharmacyApp:
    def __init__(self):
        # Initialize logging early for Streamlit runs (safe to call multiple times)
        try:
            setup_logging()
        except Exception:
            pass
        # Import modules here to avoid module-level Streamlit calls
        try:
            from core.verification_controller import VerificationController
            from core.settings_manager import SettingsManager
            from core.ocr_provider import preload_ocr_provider
            self.VerificationController = VerificationController
            self.SettingsManager = SettingsManager
        except ImportError as e:
            st.error(f"Could not import required modules: {e}")
            st.info("Please ensure verification_controller.py and settings_manager.py are in the same directory.")
            st.stop()
            
        self.config_file = "config/config.json"
        self.log_file = "verification.log"
        self.settings_manager = self.SettingsManager(self.config_file)
        self.config = None
        self.load_config()
        
        # Note: OCR provider is preloaded by VerificationController, no need to preload here
        
        # Initialize session state
        if 'verification_running' not in st.session_state:
            st.session_state.verification_running = False
        if 'verification_controller' not in st.session_state:
            st.session_state.verification_controller = None

    def load_config(self) -> bool:
        """Load configuration from config.json"""
        if self.settings_manager.load_config():
            self.config = self.settings_manager.config
            return True
        else:
            st.error(f"Configuration file {self.config_file} not found!")
            return False

    def save_config(self) -> bool:
        """Save configuration to config.json"""
        if self.config:
            self.settings_manager.config = self.config
            return self.settings_manager.save_config(create_backup=False)
        return False

    def coordinate_setup_page(self):
        """GUI Settings page with Launch GUI at top"""
        st.title("⚙️ GUI Settings")
        
        # Launch Settings GUI at the top (most important action)
        st.subheader("🚀 Launch Settings GUI")
        st.info("🎯 **Use the proven GUI tool for easy coordinate setup**")
        
        if st.button("🔧 Open Coordinate Setup Tool", type="primary"):
            try:
                subprocess.run([sys.executable, "ui/settings_gui.py"], cwd=os.getcwd())
                st.success("✅ Settings GUI launched successfully!")
            except Exception as e:
                st.error(f"❌ Failed to launch GUI: {e}")
                st.info("💡 Try running manually: `python ui/settings_gui.py`")
        
        st.markdown("---")
        
        if not self.config:
            st.error("Configuration not loaded!")
            return
        
        # Show current configuration status
        st.subheader("📋 Current Configuration Status")
        
        regions = self.config.get('regions', {})
        
        # Check if configuration is complete
        required_regions = ['trigger', 'rx_number']
        required_fields = ['patient_name', 'prescriber_name', 'drug_name', 'direction_sig']
        
        # Add enabled optional fields to the list of fields to check for completeness
        enabled_optional_fields = self.config.get("optional_fields_enabled", {})
        fields_to_check = required_fields + [field for field, enabled in enabled_optional_fields.items() if enabled]

        required_field_types = ['entered', 'source']
        
        config_complete = True
        missing_items = []
        
        # Check required regions
        for region_name in required_regions:
            if region_name not in regions:
                config_complete = False
                region_display = region_name.replace('_', ' ').title()
                missing_items.append(f"🎯 {region_display} region")
        
        # Check fields
        fields = regions.get('fields', {})
        for field_name in fields_to_check:
            field_display = field_name.replace('_', ' ').title()
            
            if field_name not in fields:
                config_complete = False
                missing_items.append(f"📝 {field_display} (both entered and source)")
            else:
                field_config = fields[field_name]
                entered_ok = 'entered' in field_config
                source_ok = 'source' in field_config
                
                if not (entered_ok and source_ok):
                    config_complete = False
                    if not entered_ok:
                        missing_items.append(f"📝 {field_display} (entered)")
                    if not source_ok:
                        missing_items.append(f"📋 {field_display} (source)")
        
        # Show completion status
        if config_complete:
            st.success("✅ **Configuration Complete!** All regions are set up and ready to use.")
        else:
            st.error("❌ **Configuration Incomplete** - Missing items:")
            for item in missing_items:
                st.write(f"  • {item}")
            
            # Show what's configured so far (only when there are missing items)
            st.subheader("📋 Current Status")
            
            # Check regions
            for region_name in required_regions:
                region_display = region_name.replace('_', ' ').title()
                if region_name in regions:
                    st.success(f"✅ **{region_display} region:** Configured")
                else:
                    st.warning(f"⚠️ **{region_display} region:** Not configured")
            
            # Check fields
            for field_name in fields_to_check:
                field_display = field_name.replace('_', ' ').title()
                
                if field_name in fields:
                    field_config = fields[field_name]
                    entered_ok = 'entered' in field_config
                    source_ok = 'source' in field_config
                    
                    if entered_ok and source_ok:
                        st.success(f"✅ **{field_display}:** Fully configured")
                    else:
                        st.warning(f"⚠️ **{field_display}:** Partially configured")
                else:
                    st.warning(f"⚠️ **{field_display}:** Not configured")
        
        st.markdown("---")
        
        # OCR Engine Settings
        self.show_ocr_engine_settings()
        
        st.markdown("---")
        
        # Trigger Detection Settings
        self.show_trigger_detection_settings()

        st.markdown("---")
        
        # Optional Fields Settings
        self.show_optional_fields_settings()
        
        st.markdown("---")
        
        # GUI launcher section
        st.subheader("🛠️ Launch Settings GUI")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.info("""
            **The Settings GUI provides:**
            • 📸 Easy screenshot capture
            • 🖱️ Click-and-drag region selection  
            • 🔍 Live OCR testing
            • 📐 Visual coordinate helpers
            • 💾 Automatic saving
            """)
        
        with col2:
            if st.button("🛠️ Launch Settings GUI", type="primary", use_container_width=True):
                self.launch_settings_gui()
        
        # Alternative methods
        st.subheader("🔧 Alternative Setup Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**📝 Manual Configuration File Editing:**")
            if st.button("📄 Open Config Folder", use_container_width=True):
                self.open_config_folder()
            st.caption("Edit config.json directly if you know the coordinates")
        
        with col2:
            st.write("**🔄 Import/Export:**")
            uploaded_file = st.file_uploader("📥 Import config.json", type="json")
            if uploaded_file is not None:
                try:
                    config_data = json.loads(uploaded_file.getvalue().decode("utf-8"))
                    self.config = config_data
                    if self.save_config():
                        st.success("✅ Configuration imported successfully!")
                        st.rerun()
                    else:
                        st.error("❌ Failed to save imported configuration!")
                except Exception as e:
                    st.error(f"❌ Import failed: {e}")
        
        # Export current config
        if st.button("📤 Export Current Configuration"):
            try:
                config_str = json.dumps(self.config, indent=2)
                st.download_button(
                    label="📄 Download config.json",
                    data=config_str,
                    file_name=f"config_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"❌ Export failed: {e}")
        
        # Configuration validation
        st.subheader("🔍 Configuration Validation")
        if st.button("✅ Validate Current Configuration"):
            self.validate_configuration()

    def launch_settings_gui(self):
        """Launch the existing settings GUI"""
        try:
            # Get the path to settings_gui.py
            gui_script = os.path.join(os.path.dirname(__file__), "settings_gui.py")
            
            if not os.path.exists(gui_script):
                st.error("❌ settings_gui.py not found in the current directory!")
                return
            
            # Launch the GUI in a separate process
            if sys.platform.startswith('win'):
                # Windows
                subprocess.Popen([sys.executable, gui_script], 
                               creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                # macOS/Linux
                subprocess.Popen([sys.executable, gui_script])
            
            st.success("✅ **Settings GUI launched!** Check for a new window.")
            st.info("💡 **Tip:** After making changes in the GUI, refresh this page to see the updated configuration.")
            
            # Add a refresh button
            if st.button("🔄 Refresh Configuration", type="secondary"):
                # Reload config from file
                self.load_config()
                st.success("✅ Configuration reloaded!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Failed to launch Settings GUI: {e}")
            st.info("💡 **Alternative:** Run `python settings_gui.py` in a terminal.")

    def open_config_folder(self):
        """Open the folder containing the configuration files"""
        try:
            folder_path = os.path.dirname(os.path.abspath(self.config_file))
            
            if sys.platform.startswith('win'):
                # Windows
                subprocess.run(['explorer', folder_path])
            elif sys.platform.startswith('darwin'):
                # macOS
                subprocess.run(['open', folder_path])
            else:
                # Linux
                subprocess.run(['xdg-open', folder_path])
                
            st.success(f"📁 Opened folder: {folder_path}")
            
        except Exception as e:
            st.error(f"❌ Failed to open folder: {e}")
            st.info(f"📁 **Manual path:** {os.path.dirname(os.path.abspath(self.config_file))}")

    def validate_configuration(self):
        """Validate the current configuration"""
        try:
            validation_results = self.settings_manager.validate_coordinates()
            
            if validation_results.get('valid', False):
                st.success("✅ **Configuration is valid!** All regions are properly set up.")
            else:
                st.error("❌ **Configuration has issues:**")
                for issue in validation_results.get('issues', []):
                    st.warning(f"⚠️ {issue}")
                    
            st.info(f"🔍 Validated {validation_results.get('regions_checked', 0)} regions")
            
        except Exception as e:
            st.error(f"❌ Validation failed: {e}")
            st.info("⚠️ This might indicate a problem with your configuration file.")

    def show_ocr_engine_settings(self):
        """Display and allow editing of OCR engine settings"""
        st.subheader("🔍 OCR Engine Settings")
        
        if not self.config:
            st.warning("No configuration loaded. Please set up coordinates first.")
            return
        
        # Check available OCR engines
        available_engines = self.check_available_ocr_engines()
        
        # Get current OCR provider
        current_provider = self.config.get("ocr_provider", "tesseract")
        
        st.info("💡 **Global OCR Engine:** This setting applies to ALL verification methods (Local OCR, LLM AI, etc.). Choose the best OCR engine for your CPU performance needs.")
        
        # Display OCR engine information
        engine_info = {
            "auto": {"name": "Auto (Recommended)", "speed": "Optimal", "accuracy": "95%", "description": "Smart selection based on available hardware"},
            "easyocr": {"name": "EasyOCR", "speed": "Good", "accuracy": "92%", "description": "High accuracy, GPU-accelerated"},
            "tesseract": {"name": "Tesseract", "speed": "Fast", "accuracy": "85%", "description": "CPU-optimized, widely compatible"},
            "paddleocr": {"name": "PaddleOCR", "speed": "Good", "accuracy": "89%", "description": "Comprehensive text detection, CPU optimized"}
        }
        
        # Create OCR engine selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**Available OCR Options:**")
            
            # Display engines with status
            for engine_key, info in engine_info.items():
                if engine_key == "auto":
                    is_available = True  # Auto is always available
                else:
                    is_available = available_engines.get(engine_key, False)
                is_current = (engine_key == current_provider)
                
                status_icon = "✅" if is_available else "❌"
                current_icon = "🎯" if is_current else "  "
                
                col_a, col_b, col_c, col_d = st.columns([0.3, 0.3, 1.5, 1])
                
                with col_a:
                    st.write(status_icon)
                with col_b:
                    st.write(current_icon)
                with col_c:
                    st.write(f"**{info['name']}**")
                with col_d:
                    st.write(f"{info['speed']}, {info['accuracy']}")
                
                if not is_available:
                    install_commands = {
                        "easyocr": "pip install easyocr",
                        "tesseract": "pip install pytesseract (also requires Tesseract binary)",
                        "paddleocr": "pip install paddlepaddle paddleocr"
                    }
                    cmd = install_commands.get(engine_key, f"pip install {engine_key}")
                    st.caption(f"   💡 Install with: `{cmd}`")
        
        with col2:
            st.write("**Select OCR Engine:**")
            
            # Get available engine options for selection
            available_options = []
            available_labels = []
            
            for engine_key, info in engine_info.items():
                if engine_key == "auto" or available_engines.get(engine_key, False):
                    available_options.append(engine_key)
                    available_labels.append(f"{info['name']} ({info['accuracy']})")
            
            if available_options:
                try:
                    current_index = available_options.index(current_provider) if current_provider in available_options else 0
                except ValueError:
                    current_index = 0
                
                selected_engine = st.selectbox(
                    "Choose OCR Engine:",
                    options=available_options,
                    format_func=lambda x: f"{engine_info[x]['name']} ({engine_info[x]['accuracy']})",
                    index=current_index,
                    help="Select the OCR engine to use for text recognition"
                )
                
                # Update OCR provider if changed
                if selected_engine != current_provider:
                    self.config["ocr_provider"] = selected_engine
                    
                    # Save the configuration
                    try:
                        self.settings_manager.config = self.config
                        self.settings_manager.save_config(create_backup=True)
                        st.success(f"✅ OCR engine updated to {engine_info[selected_engine]['name']}!")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to save OCR engine setting: {e}")
            else:
                st.error("❌ No OCR engines available!")
                st.info("Install at least one OCR engine to continue.")
        
        # OCR Engine-specific settings
        if current_provider in available_engines and available_engines[current_provider]:
            st.write("**Engine-Specific Settings:**")
            
            if current_provider == "easyocr":
                self.show_easyocr_settings()
            elif current_provider == "tesseract":
                self.show_tesseract_settings()
            elif current_provider == "paddleocr":
                self.show_paddleocr_settings()
    
    def check_available_ocr_engines(self):
        """Check which OCR engines are available"""
        engines = {
            'easyocr': False,
            'tesseract': False,
            'paddleocr': False
        }
        
        # Check EasyOCR
        try:
            import easyocr
            engines['easyocr'] = True
        except ImportError:
            pass
        
        # Check Tesseract
        try:
            import pytesseract
            engines['tesseract'] = True
        except ImportError:
            pass
        
        # Check PaddleOCR
        try:
            import paddleocr
            engines['paddleocr'] = True
        except ImportError:
            pass
        
        return engines
    
    def show_easyocr_settings(self):
        """Show EasyOCR-specific settings"""
        easyocr_config = self.config.get("easyocr", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_gpu = st.checkbox(
                "🚀 Use GPU acceleration",
                value=easyocr_config.get("use_gpu", True),
                help="Enable GPU acceleration for faster processing"
            )
        
        with col2:
            confidence = st.slider(
                "🎯 Confidence threshold",
                min_value=0.1,
                max_value=1.0,
                value=easyocr_config.get("confidence_threshold", 0.5),
                step=0.1,
                help="Minimum confidence score for OCR results"
            )
        
        # Update settings if changed
        new_easyocr_config = {
            "use_gpu": use_gpu,
            "confidence_threshold": confidence
        }
        
        if new_easyocr_config != easyocr_config:
            self.config["easyocr"] = new_easyocr_config
            self.save_config_changes()
    
    def show_tesseract_settings(self):
        """Show Tesseract-specific settings"""
        tesseract_config = self.config.get("tesseract", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            config_options = st.text_input(
                "🔧 Config options",
                value=tesseract_config.get("config_options", "--psm 7"),
                help="Tesseract configuration options (e.g., --psm 7)"
            )
        
        with col2:
            fallback_config = st.text_input(
                "🔄 Fallback config",
                value=tesseract_config.get("fallback_config", "--psm 8"),
                help="Fallback configuration if primary config fails"
            )
        
        # Update settings if changed
        new_tesseract_config = {
            "config_options": config_options,
            "fallback_config": fallback_config
        }
        
        if new_tesseract_config != tesseract_config:
            self.config["tesseract"] = new_tesseract_config
            self.save_config_changes()
    
    def save_config_changes(self):
        """Save configuration changes with error handling"""
        try:
            self.settings_manager.config = self.config
            self.settings_manager.save_config(create_backup=True)
            st.success("✅ Settings updated!")
            time.sleep(0.5)
            st.rerun()
        except Exception as e:
            st.error(f"❌ Failed to save settings: {e}")
    
    def show_paddleocr_settings(self):
        """Show PaddleOCR-specific settings"""
        paddleocr_config = self.config.get("paddleocr", {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            use_gpu = st.checkbox(
                "🚀 Use GPU acceleration",
                value=paddleocr_config.get("use_gpu", False),
                help="Enable GPU acceleration (requires CUDA-compatible GPU)"
            )
        
        with col2:
            confidence_threshold = st.slider(
                "🎯 Confidence threshold",
                min_value=0.1,
                max_value=1.0,
                value=paddleocr_config.get("confidence_threshold", 0.5),
                step=0.1,
                help="Minimum confidence score for OCR results"
            )
        
        use_angle_cls = st.checkbox(
            "🔄 Use text angle classification",
            value=paddleocr_config.get("use_angle_cls", True),
            help="Enable text rotation detection and correction"
        )
        
        show_log = st.checkbox(
            "📝 Show PaddleOCR logs",
            value=paddleocr_config.get("show_log", False),
            help="Display PaddleOCR internal logging messages"
        )
        
        lang_options = ['en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko']
        current_lang = paddleocr_config.get("lang", "en")
        lang = st.selectbox(
            "🌐 Language",
            options=lang_options,
            index=lang_options.index(current_lang) if current_lang in lang_options else 0,
            help="Primary language for OCR recognition"
        )
        
        # Update settings if changed
        new_paddleocr_config = {
            "use_gpu": use_gpu,
            "confidence_threshold": confidence_threshold,
            "use_angle_cls": use_angle_cls,
            "show_log": show_log,
            "lang": lang
        }
        
        if new_paddleocr_config != paddleocr_config:
            self.config["paddleocr"] = new_paddleocr_config
            self.save_config_changes()

    def show_trigger_detection_settings(self):
        """Display and allow editing of trigger detection settings"""
        st.subheader("🎯 Trigger Detection Settings")
        
        if not self.config:
            st.warning("No configuration loaded. Please set up coordinates first.")
            return
            
        st.info("💡 **Trigger Settings:** Configure what keywords and matching criteria to use for detecting new prescriptions.")
        
        # Get current advanced settings
        advanced_settings = self.config.get("advanced_settings", {})
        trigger_config = advanced_settings.get("trigger", {})
        
        # Keywords settings
        st.write("**🔍 Detection Keywords:**")
        current_keywords = trigger_config.get("keywords", ["pre", "check", "rx"])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            keywords_text = st.text_input(
                "Keywords (comma-separated):",
                value=", ".join(current_keywords),
                help="Words to look for in the trigger area (e.g., 'pre, check, rx')"
            )
            st.caption("🔎 These words will trigger prescription verification")
        
        with col2:
            min_matches = st.number_input(
                "Minimum keyword matches:",
                min_value=1,
                max_value=10,
                value=trigger_config.get("min_keyword_matches", 2),
                step=1,
                help="How many keywords must match to trigger verification"
            )
            st.caption("🎯 Required matches to activate")
        
        # Matching settings
        st.write("**⚙️ Matching Settings:**")
        
        col3, col4 = st.columns(2)
        
        with col3:
            similarity_threshold = st.slider(
                "🎯 Keyword similarity threshold",
                min_value=50,
                max_value=100,
                value=trigger_config.get("keyword_similarity_threshold", 90),
                step=5,
                help="How closely words must match keywords (higher = more strict)"
            )
            st.caption("📊 90%+ recommended for accuracy")
        
        with col4:
            reset_delay = st.number_input(
                "🔄 Reset delay (seconds)",
                min_value=1.0,
                max_value=30.0,
                value=trigger_config.get("lost_reset_delay_seconds", 5.0),
                step=1.0,
                help="How long to wait before resetting when trigger is lost"
            )
            st.caption("⏱️ Time before accepting new prescriptions")
        
        # Show current configuration summary
        st.write("**📋 Current Trigger Configuration:**")
        keywords_list = [kw.strip() for kw in keywords_text.split(",") if kw.strip()]
        config_summary = f"Keywords: {keywords_list} | Min matches: {min_matches} | Similarity: {similarity_threshold}% | Reset: {reset_delay}s"
        st.code(config_summary)
        
        # Update settings if changed
        new_trigger_config = {
            "keywords": keywords_list,
            "keyword_similarity_threshold": similarity_threshold,
            "min_keyword_matches": min_matches,
            "lost_reset_delay_seconds": reset_delay
        }
        
        if new_trigger_config != trigger_config:
            if "advanced_settings" not in self.config:
                self.config["advanced_settings"] = {}
            if "trigger" not in self.config["advanced_settings"]:
                self.config["advanced_settings"]["trigger"] = {}
            
            self.config["advanced_settings"]["trigger"].update(new_trigger_config)
            self.save_config_changes()

    def show_optional_fields_settings(self):
        """Display and allow editing of optional fields to verify."""
        with st.expander("➕ Optional Verification Fields (Local OCR+Fuzzy only)", expanded=False):
            if not self.config:
                st.warning("No configuration loaded.")
                return

            st.info("💡 Select additional fields to verify using Local OCR+Fuzzy comparison. You must set up their coordinates after enabling them.")

            optional_fields_config = {
                "patient_dob": "Patient DOB",
                "patient_address": "Patient Address",
                "patient_phone": "Patient Phone",
                "prescriber_address": "Prescriber Address"
            }

            enabled_fields = self.config.get("optional_fields_enabled", {})
            changes_made = False
            
            col1, col2 = st.columns(2)
            
            for i, (key, label) in enumerate(optional_fields_config.items()):
                col = col1 if i % 2 == 0 else col2
                with col:
                    current_value = enabled_fields.get(key, False)
                    new_value = st.checkbox(label, value=current_value, key=f"optional_{key}")

                    if new_value != current_value:
                        if "optional_fields_enabled" not in self.config:
                            self.config["optional_fields_enabled"] = {}
                        self.config["optional_fields_enabled"][key] = new_value
                        changes_made = True

                        # If a field is enabled, add a placeholder config for it
                        if new_value:
                            if "fields" not in self.config["regions"]:
                                self.config["regions"]["fields"] = {}
                            if key not in self.config["regions"]["fields"]:
                                threshold_key = key
                                self.config["regions"]["fields"][key] = {
                                    "entered": [0, 0, 0, 0],
                                    "source": [0, 0, 0, 0],
                                    "score_fn": "ratio",
                                    "threshold_key": threshold_key
                                }
                                if "thresholds" not in self.config:
                                    self.config["thresholds"] = {}
                                if threshold_key not in self.config["thresholds"]:
                                    self.config["thresholds"][threshold_key] = 65

            if changes_made:
                self.save_config_changes()

    def show_threshold_settings(self):
        """Display and allow editing of score thresholds"""
        st.subheader("🎯 Score Thresholds")
        
        if not self.config:
            st.warning("No configuration loaded. Please set up coordinates first.")
            return
            
        # Get current thresholds
        thresholds = self.config.get("thresholds", {})
        
        st.info("💡 **Threshold Settings:** Higher values require closer matches. Lower values are more lenient.")
        
        # Create threshold controls
        col1, col2 = st.columns(2)
        
        threshold_fields = [
            ("patient", "👤 Patient Name"),
            ("prescriber", "👨‍⚕️ Prescriber"),
            ("drug", "💊 Drug Name"),
            ("sig", "📝 Sig/Instructions")
        ]
        
        # Add optional fields if they are enabled
        enabled_optional_fields = self.config.get("optional_fields_enabled", {})
        optional_fields_map = {
            "patient_dob": "🎂 Patient DOB",
            "patient_address": "🏠 Patient Address",
            "patient_phone": "📞 Patient Phone",
            "prescriber_address": "🏥 Prescriber Address"
        }

        for key, label in optional_fields_map.items():
            if enabled_optional_fields.get(key, False):
                threshold_fields.append((key, label))
        
        changes_made = False
        new_thresholds = {}
        
        for i, (key, label) in enumerate(threshold_fields):
            # Alternate between columns
            col = col1 if i % 2 == 0 else col2
            
            with col:
                current_value = thresholds.get(key, 65)
                new_value = st.slider(
                    label,
                    min_value=30,
                    max_value=95,
                    value=current_value,
                    step=5,
                    help=f"Match threshold for {label.lower()} (currently {current_value}%)"
                )
                
                new_thresholds[key] = new_value
                if new_value != current_value:
                    changes_made = True
        
        # Show current threshold summary
        st.write("**Current Thresholds:**")
        threshold_summary = " | ".join([f"{label}: {new_thresholds[key]}%" for key, label in threshold_fields])
        st.code(threshold_summary)
        
        # Save changes if any
        if changes_made:
            self.config["thresholds"] = new_thresholds
            try:
                # Update the settings manager's config and save
                self.settings_manager.config = self.config
                self.settings_manager.save_config(create_backup=True)
                st.success("✅ Threshold settings updated!")
                time.sleep(0.5)  # Brief pause to show the message
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save threshold settings: {e}")

    def show_verification_method_settings(self):
        """Display and allow editing of verification methods for each field."""
        st.subheader("🤖 Verification Methods")
        
        if not self.config:
            st.warning("No configuration loaded.")
            return

        st.info("💡 Choose the verification method for each field. 'AI' is available for local LLMs without an API key.")

        fields_config = self.config.get("regions", {}).get("fields", {})

        # Get all fields to show
        field_map = {
            "patient_name": "👤 Patient Name",
            "prescriber_name": "👨‍⚕️ Prescriber",
            "drug_name": "💊 Drug Name",
            "direction_sig": "📝 Sig/Instructions",
            "patient_dob": "🎂 Patient DOB",
            "patient_address": "🏠 Patient Address",
            "patient_phone": "📞 Patient Phone",
            "prescriber_address": "🏥 Prescriber Address"
        }
        
        mandatory_fields = ["patient_name", "prescriber_name", "drug_name", "direction_sig"]
        enabled_optional_fields = [k for k, v in self.config.get("optional_fields_enabled", {}).items() if v]
        all_fields = mandatory_fields + enabled_optional_fields

        changes_made = False
        
        col1, col2 = st.columns(2)
        
        for i, field_name in enumerate(all_fields):
            col = col1 if i % 2 == 0 else col2
            with col:
                field_label = field_map.get(field_name, field_name.replace('_', ' ').title())
                current_method = fields_config.get(field_name, {}).get("verification_method", "fuzzy")
                
                options = ["fuzzy", "ai"]
                
                # Ensure current_method is valid
                if current_method not in options:
                    current_method = "fuzzy"

                try:
                    current_index = options.index(current_method)
                except ValueError:
                    current_index = 0

                new_method = st.selectbox(
                    field_label,
                    options=options,
                    index=current_index,
                    key=f"method_{field_name}"
                )

                if new_method != current_method:
                    if field_name not in self.config.get("regions", {}).get("fields", {}):
                        self.config["regions"]["fields"][field_name] = {}
                    self.config["regions"]["fields"][field_name]["verification_method"] = new_method
                    changes_made = True

        if changes_made:
            self.save_config_changes()

    def monitoring_page(self):
        """Display the monitoring/logging page with live OCR and scores"""
        st.title("🏠 Pharmacy Verification - Main Dashboard")

        # Verification Method Selection
        st.subheader("🔧 Verification Method")
        self.show_verification_method_selection()
        
        st.markdown("---")

        # Controls at the top
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            if st.button("🟢 Start Monitoring", disabled=st.session_state.verification_running):
                self.start_verification()
        with c2:
            if st.button("🔴 Stop Monitoring", disabled=not st.session_state.verification_running):
                self.stop_verification()
        with c3:
            status_color = "🟢" if st.session_state.verification_running else "🔴"
            status_text = "Running" if st.session_state.verification_running else "Stopped"
            st.markdown(f"**Status:** {status_color} {status_text}")

        st.markdown("---")

        # Automation settings (kept as-is)
        self.show_automation_settings()

        # Score threshold settings (moved from GUI settings page)
        with st.expander("🎯 Score Thresholds", expanded=False):
            self.show_threshold_settings()

        # Timing settings hidden by default in an expander
        with st.expander("⏱️ Timing Settings (advanced)", expanded=False):
            self.show_timing_settings()

        # Live Terminal Output Display (monitoring only)
        if st.session_state.verification_running:
            self.show_live_terminal_output()

    def show_verification_method_selection(self):
        """Show verification method selection with 3 options: Local OCR+Fuzzy, LLM AI, and VLM AI."""
        if not self.config:
            return

        # Handle legacy configuration
        current_method = self.config.get("verification_method", "local_ocr_fuzzy")
        if "verification_mode" in self.config and "verification_method" not in self.config:
            legacy_mode = self.config.get("verification_mode", "ocr")
            current_method = "vlm_ai" if legacy_mode == "vlm" else "local_ocr_fuzzy"

        # Method options with descriptions
        method_options = [
            ("local_ocr_fuzzy", "� Local OCR + Fuzzy Matching", "Multiple small OCR areas + CPU-based text comparison"),
            ("vlm_ai", "👁️ VLM AI (Direct Vision)", "Single image analysis with prompt engineering (no OCR)")
        ]
        
        method_values = [value for value, _, _ in method_options]
        method_labels = {value: f"{label}" for value, label, _ in method_options}
        method_descriptions = {value: desc for value, _, desc in method_options}

        try:
            current_index = method_values.index(current_method)
        except ValueError:
            current_index = 0

        selected_method = st.selectbox(
            "Verification Method:",
            options=method_values,
            format_func=lambda x: method_labels[x],
            index=current_index,
            help="Choose how prescription fields are verified"
        )

        # Show description of selected method
        if selected_method in method_descriptions:
            st.info(f"📝 **{method_labels[selected_method]}**: {method_descriptions[selected_method]}")

        # Show method-specific requirements and configuration hints
        self._show_method_requirements(selected_method)

        # Save configuration
        if selected_method != current_method:
            self.config["verification_method"] = selected_method
            # Clean up legacy configuration
            if "verification_mode" in self.config:
                del self.config["verification_mode"]
            try:
                self.settings_manager.config = self.config
                self.settings_manager.save_config(create_backup=True)
                st.success("✅ Verification method updated!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save verification method: {e}")

    def _show_method_requirements(self, method: str):
        """Show requirements and configuration hints for selected verification method."""
        if method == "local_ocr_fuzzy":
            st.success("✅ **Ready to use** - All required components available")
            st.info("💡 **Setup**: Configure individual field regions using the Settings GUI")
            
        elif method == "vlm_ai":
            # Check if VLM config exists and has regions with coordinates
            vlm_config_exists = os.path.exists("config/vlm_config.json")
            vlm_regions_exist = False
            
            if vlm_config_exists:
                try:
                    with open("config/vlm_config.json", 'r', encoding='utf-8') as f:
                        vlm_config = json.load(f)
                    # Check if vlm_regions exists and has any region with coordinates
                    vlm_regions = vlm_config.get("vlm_regions", {})
                    if vlm_regions and any(isinstance(coords, list) and len(coords) == 4 for coords in vlm_regions.values()):
                        vlm_regions_exist = True
                except Exception as e:
                    pass
            
            # Only show warnings if something is missing
            if vlm_config_exists and vlm_regions_exist:
                st.success("✅ **Ready to use** - All required components available")
                st.info("💡 **Setup**: Configure VLM settings using the VLM Configuration page")
            else:
                st.warning("⚠️ **Configuration required**:")
                if not vlm_config_exists:
                    st.write("• Configure VLM API settings in **VLM Configuration** page")
                if not vlm_regions_exist:
                    st.write("• Set up image region and prompts using **VLM Configuration** page")

    def show_automation_settings(self):
        """Display automation settings section"""
        st.subheader("🤖 Automation Settings")
        
        if not self.config:
            st.warning("No configuration loaded. Please set up coordinates first.")
            return
            
        # Get current automation settings
        automation_config = self.config.get("automation", {})
        current_enabled = automation_config.get("send_key_on_all_match", False)
        current_key = automation_config.get("key_on_all_match", "f12")
        current_delay = automation_config.get("key_delay_seconds", 0.5)
        
        # Create automation controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            enabled = st.checkbox(
                "🎯 Send key when all fields match", 
                value=current_enabled,
                help="Automatically send a key press when all verification fields match"
            )
        
        with col2:
            key_options = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "enter", "tab", "space", "escape"]
            selected_key = st.selectbox(
                "Key to send:",
                options=key_options,
                index=key_options.index(current_key) if current_key in key_options else key_options.index("f12")
            )
        
        with col3:
            delay = st.number_input(
                "Delay (seconds):",
                min_value=0.1,
                max_value=5.0,
                value=current_delay,
                step=0.1
            )
        
        # Update automation settings if changed
        if enabled != current_enabled or selected_key != current_key or delay != current_delay:
            if "automation" not in self.config:
                self.config["automation"] = {}
            
            self.config["automation"]["send_key_on_all_match"] = enabled
            self.config["automation"]["key_on_all_match"] = selected_key
            self.config["automation"]["key_delay_seconds"] = delay
            
            # Save the configuration
            try:
                # Update the settings manager's config first
                self.settings_manager.config = self.config
                self.settings_manager.save_config(create_backup=True)
                st.success("✅ Automation settings updated!")
                time.sleep(0.5)  # Brief pause to show the message
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save automation settings: {e}")

    def show_timing_settings(self):
        """Display timing settings section"""
        st.subheader("⏱️ Timing Settings")
        
        if not self.config:
            st.warning("No configuration loaded. Please set up coordinates first.")
            return
            
        # Get current timing settings
        timing_config = self.config.get("timing", {})
        
        st.info("💡 **Timing Settings:** Control how fast the system polls for changes and waits between operations.")
        
        # Create timing controls in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Polling & Detection:**")
            
            fast_polling = st.number_input(
                "🔄 Fast polling - Active monitoring",
                min_value=0.1,
                max_value=5.0,
                value=timing_config.get("fast_polling_seconds", 0.2),
                step=0.1,
                help="How often to check for screen changes when active (lower = more responsive but uses more CPU)"
            )
            st.caption("⚡ How quickly the system detects new prescriptions")
            
            same_rx_wait = st.number_input(
                "🔄 Same Rx cooldown - Duplicate prevention",
                min_value=1.0,
                max_value=30.0,
                value=timing_config.get("same_prescription_wait_seconds", 3.0),
                step=1.0,
                help="Prevent duplicate processing of the same prescription (higher = less likely to re-process same Rx)"
            )
            st.caption("🚫 Prevents processing the same prescription twice")
        
        with col2:
            st.write("**Verification Delays:**")
            
            trigger_content_delay = st.number_input(
                "⚡ Trigger content load - Wait for UI",
                min_value=0.0,
                max_value=5.0,
                value=timing_config.get("trigger_content_load_delay_seconds", 0.5),
                step=0.1,
                help="Wait after trigger detection for content to fully load (increase if fields appear slowly)"
            )
            st.caption("⏳ Lets the screen finish loading after detection")
            
            trigger_check_interval = st.number_input(
                "🎯 Trigger check interval - Detection frequency",
                min_value=0.1,
                max_value=5.0,
                value=timing_config.get("trigger_check_interval_seconds", 1.0),
                step=0.1,
                help="How often to check for trigger text (lower = faster detection but more CPU usage)"
            )
            st.caption("🔍 How often to look for the trigger")
        
        # Show current timing summary
        st.write("**Current Timing Configuration:**")
        timing_summary = f"Fast Poll: {fast_polling}s | Content Load: {trigger_content_delay}s | Trigger Check: {trigger_check_interval}s | Rx Cooldown: {same_rx_wait}s"
        st.code(timing_summary)
        
        # Update timing settings
        new_timing_config = {
            "fast_polling_seconds": fast_polling,
            "same_prescription_wait_seconds": same_rx_wait,
            "trigger_content_load_delay_seconds": trigger_content_delay,
            "trigger_check_interval_seconds": trigger_check_interval
        }
        
        # Check if settings changed
        if new_timing_config != timing_config:
            self.config["timing"] = new_timing_config
            
            # Save the configuration
            try:
                self.settings_manager.config = self.config
                self.settings_manager.save_config(create_backup=True)
                st.success("✅ Timing settings updated!")
                time.sleep(0.5)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save timing settings: {e}")

    def show_live_terminal_output(self):
        """Display live terminal output from the verification process"""
        st.subheader("�️ Live Terminal Output")
        
        # Check if verification is actually running
        if not st.session_state.verification_running:
            st.warning("❌ Verification not running - start monitoring to see terminal output")
            return
        
        # Add refresh controls
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🔄 Refresh Output", key="refresh_terminal"):
                st.rerun()
        
        with col2:
            auto_refresh = st.checkbox("🔄 Auto-refresh (3s)", key="auto_refresh_terminal")
        
        with col3:
            show_lines = st.selectbox("Show lines:", [50, 100, 200, 500], index=1, key="terminal_lines")
        
        # Auto-refresh functionality
        if auto_refresh:
            time.sleep(3)
            st.rerun()
        
        # Read the log file for live output
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                if log_content.strip():
                    # Get the last N lines
                    lines = log_content.strip().split('\n')
                    recent_lines = lines[-show_lines:] if len(lines) > show_lines else lines
                    
                    # Display with syntax highlighting for better readability
                    terminal_output = '\n'.join(recent_lines)
                    
                    # Create a container for the terminal output
                    with st.container():
                        st.markdown("**� Real-time Verification Output:**")
                        
                        # Show the terminal-like output
                        st.text_area(
                            "Terminal Output",
                            terminal_output,
                            height=500,
                            key=f"terminal_output_{int(time.time())}"
                        )
                        
                        # Show status info
                        total_lines = len(lines)
                        st.caption(f"📈 Showing last {len(recent_lines)} of {total_lines} total lines | Last updated: {datetime.now().strftime('%H:%M:%S')}")
                        
                        # Add scroll to bottom hint
                        if len(lines) > show_lines:
                            st.info(f"💡 Showing most recent {show_lines} lines. Increase the line count to see more history.")
                else:
                    st.info("🕐 Waiting for verification output... Make sure monitoring is active.")
                    
            else:
                st.warning("📄 No log file found. The verification process may not have started yet.")
                st.info("💡 Click 'Start Monitoring' to begin capturing terminal output.")
                
        except Exception as e:
            st.error(f"❌ Error reading terminal output: {e}")
            st.info("This might be a temporary file access issue. Try refreshing.")
        
        # Add a section for quick stats if there's content
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if content.strip():
                    # Quick stats
                    lines = content.strip().split('\n')
                    
                    # Count different types of log entries
                    debug_count = sum(1 for line in lines if 'DEBUG' in line.upper())
                    match_count = sum(1 for line in lines if 'Match: True' in line)
                    error_count = sum(1 for line in lines if 'ERROR' in line.upper() or 'Error:' in line)
                    
                    # Show stats in columns
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("📊 Total Lines", len(lines))
                    
                    with stat_col2:
                        st.metric("✅ Matches Found", match_count)
                    
                    with stat_col3:
                        st.metric("🔍 Debug Entries", debug_count)
                    
                    with stat_col4:
                        st.metric("❌ Errors", error_count)
                        
            except Exception:
                pass  # Skip stats if there's an issue

    def show_log_section(self):
        """Display log section"""
        with st.expander("📋 Recent Activity Logs", expanded=False):
            if os.path.exists(self.log_file):
                try:
                    with open(self.log_file, 'r', encoding='utf-8') as f:
                        log_content = f.read()
                    
                    if log_content.strip():
                        # Show last 10 lines instead of 20 to save space
                        lines = log_content.strip().split('\n')[-10:]
                        st.text_area("Log Output", '\n'.join(lines), height=200)
                    else:
                        st.info("No logs yet. Start monitoring to see activity.")
                except Exception as e:
                    st.error(f"Error reading log file: {e}")
            else:
                st.info("No log file found. Start monitoring to begin collecting data.")

    def start_verification(self):
        """Start the verification process"""
        try:
            if self.config and hasattr(self, 'VerificationController'):
                # Create event loop for the controller
                loop = asyncio.new_event_loop()
                
                # Create verification controller with both config and loop
                controller = self.VerificationController(self.config, loop)
                st.session_state.verification_controller = controller
                
                # Start monitoring in a background thread
                def run_monitoring():
                    try:
                        asyncio.set_event_loop(loop)
                        loop.run_until_complete(controller.async_run())
                    except Exception as e:
                        logging.error(f"Error in monitoring thread: {e}")
                    finally:
                        loop.close()
                
                # Start the monitoring thread
                monitoring_thread = threading.Thread(target=run_monitoring, daemon=True)
                monitoring_thread.start()
                
                st.session_state.verification_running = True
                st.session_state.monitoring_thread = monitoring_thread
                
                st.success("✅ Verification monitoring started!")
                # Force UI to refresh so Start disables and status updates immediately
                st.rerun()
            else:
                st.error("Cannot start verification: Configuration not loaded!")
        except Exception as e:
            st.error(f"Failed to start verification: {e}")

    def stop_verification(self):
        """Stop the verification process"""
        try:
            st.session_state.verification_running = False
            
            if st.session_state.verification_controller:
                st.session_state.verification_controller.stop()
                st.session_state.verification_controller = None
            
            if 'monitoring_thread' in st.session_state:
                st.session_state.monitoring_thread = None
                
            st.success("🛑 Verification monitoring stopped!")
            # Force UI to refresh so Start enables again immediately
            st.rerun()
            
        except Exception as e:
            st.error(f"Error stopping verification: {e}")
            st.session_state.verification_running = False
            st.session_state.verification_controller = None

    def vlm_config_page(self):
        """Displays the VLM configuration page"""
        if self.config:
            vlm_settings_page(self.config)
        else:
            st.error("❌ Configuration not loaded")
            st.info("Please load configuration first in the Settings page")

    def run(self):
        """Main application runner"""        
        # Sidebar navigation
        st.sidebar.title("💊 Pharmacy Verification")
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["🏠 Main/Home", "⚙️ Settings - GUI", "👁️ VLM Configuration"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Quick Info")
        
        if self.config:
            st.sidebar.success("✅ Configuration loaded")
        else:
            st.sidebar.error("❌ Configuration not loaded")
        
        # Main content based on selected page
        if page == "🏠 Main/Home":
            self.monitoring_page()
        elif page == "⚙️ Settings - GUI":
            self.coordinate_setup_page()
        elif page == "👁️ VLM Configuration":
            self.vlm_config_page()


def main():
    """Main application entry point"""
    # Page configuration - must be first Streamlit call
    st.set_page_config(
        page_title="Simple Pharmacy Verification",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    try:
        # Create and run the app
        app = SimplePharmacyApp()
        app.run()
        
    except Exception as e:
        st.error(f"Failed to start application: {e}")
        st.info("Please ensure all dependencies are installed")

if __name__ == "__main__":
    main()
