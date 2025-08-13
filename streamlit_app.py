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
import time
import logging
from logger_config import setup_logging
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image, ImageDraw
import threading
import subprocess
import sys
from typing import Dict, Any, List, Optional, Tuple

class SimplePharmacyApp:
    def __init__(self):
        # Initialize logging early for Streamlit runs (safe to call multiple times)
        try:
            setup_logging()
        except Exception:
            pass
        # Import modules here to avoid module-level Streamlit calls
        try:
            from verification_controller import VerificationController
            from settings_manager import SettingsManager
            self.VerificationController = VerificationController
            self.SettingsManager = SettingsManager
        except ImportError as e:
            st.error(f"Could not import required modules: {e}")
            st.info("Please ensure verification_controller.py and settings_manager.py are in the same directory.")
            st.stop()
            
        self.config_file = "config.json"
        self.log_file = "verification.log"
        self.settings_manager = self.SettingsManager(self.config_file)
        self.config = None
        self.load_config()
        
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
        """Launch the existing GUI for coordinate setup"""
        st.title("📍 Pharmacy Verification - Coordinate Setup")
        st.info("🎯 **Use the proven GUI tool for easy coordinate setup**")
        
        if not self.config:
            st.error("Configuration not loaded!")
            return
        
        # Show current configuration status
        st.subheader("📋 Current Configuration Status")
        
        regions = self.config.get('regions', {})
        
        # Check if configuration is complete
        required_regions = ['trigger']
        required_fields = ['patient_name', 'prescriber_name', 'drug_name', 'direction_sig']
        required_field_types = ['entered', 'source']
        
        config_complete = True
        missing_items = []
        
        # Check trigger
        if 'trigger' not in regions:
            config_complete = False
            missing_items.append("🎯 Trigger region")
        else:
            st.success("✅ **Trigger region:** Configured")
        
        # Check fields
        fields = regions.get('fields', {})
        for field_name in required_fields:
            field_display = field_name.replace('_', ' ').title()
            
            if field_name not in fields:
                config_complete = False
                missing_items.append(f"📝 {field_display} (both entered and source)")
                st.warning(f"⚠️ **{field_display}:** Not configured")
            else:
                field_config = fields[field_name]
                entered_ok = 'entered' in field_config
                source_ok = 'source' in field_config
                
                if entered_ok and source_ok:
                    st.success(f"✅ **{field_display}:** Fully configured")
                else:
                    config_complete = False
                    if not entered_ok:
                        missing_items.append(f"📝 {field_display} (entered)")
                    if not source_ok:
                        missing_items.append(f"📋 {field_display} (source)")
                    st.warning(f"⚠️ **{field_display}:** Partially configured")
        
        # Show completion status
        if config_complete:
            st.success("✅ **Configuration Complete!** All regions are set up and ready to use.")
        else:
            st.error("❌ **Configuration Incomplete** - Missing items:")
            for item in missing_items:
                st.write(f"  • {item}")
        
        st.markdown("---")
        
        # Score Threshold Settings
        self.show_threshold_settings()
        
        st.markdown("---")
        
        # GUI launcher section
        st.subheader("🛠️ Launch Coordinate Setup Tool")
        
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

    def monitoring_page(self):
        """Display the monitoring/logging page with live OCR and scores"""
        st.title("📊 Pharmacy Verification Monitor")
        
        # Automation settings section
        self.show_automation_settings()
        
        # Control buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🟢 Start Monitoring", disabled=st.session_state.verification_running):
                self.start_verification()
                
        with col2:
            if st.button("🔴 Stop Monitoring", disabled=not st.session_state.verification_running):
                self.stop_verification()
                
        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()
                
        with col4:
            if st.button("🗑️ Clear Logs"):
                if os.path.exists(self.log_file):
                    open(self.log_file, 'w', encoding='utf-8').close()
                st.success("Logs cleared!")
                st.rerun()

        # Status indicator
        status_color = "🟢" if st.session_state.verification_running else "🔴"
        status_text = "Running" if st.session_state.verification_running else "Stopped"
        st.markdown(f"**Status:** {status_color} {status_text}")
        
        # Live Terminal Output Display
        if st.session_state.verification_running:
            self.show_live_terminal_output()
        
        # Log display (smaller section now)
        self.show_log_section()

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
                # Create verification controller
                controller = self.VerificationController(self.config)
                st.session_state.verification_controller = controller
                
                # Start monitoring in a background thread
                def run_monitoring():
                    try:
                        controller.run()
                    except Exception as e:
                        logging.error(f"Error in monitoring thread: {e}")
                
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

    def run(self):
        """Main application runner"""        
        # Sidebar navigation
        st.sidebar.title("💊 Pharmacy Verification")
        st.sidebar.markdown("---")
        
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["📊 Monitor & Logs", "📍 Setup Coordinates (GUI)"]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📋 Quick Info")
        
        if self.config:
            st.sidebar.success("✅ Configuration loaded")
        else:
            st.sidebar.error("❌ Configuration not loaded")
        
        # Main content based on selected page
        if page == "📊 Monitor & Logs":
            self.monitoring_page()
        elif page == "📍 Setup Coordinates (GUI)":
            self.coordinate_setup_page()

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
