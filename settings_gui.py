#!/usr/bin/env python3
"""
Enhanced Settings GUI for Pharmacy Verification System
======================================================

This tool provides a graphical interface for adjusting screen coordinates and
configuring general program settings. Features include:

- Visual rectangle overlay on screenshots
- Drag-and-drop coordinate adjustment
- Matching threshold configuration
- Automation settings management
- Import/export configuration files
- OCR testing on selected regions
- Zoom functionality and validation tools

Dependencies:
- tkinter (usually included with Python)
- pyautogui (screen capture)
- PIL/Pillow (image processing)
- pytesseract (OCR testing)

If tkinter is not available, install with:
  macOS: brew install python-tk
  Ubuntu: sudo apt-get install python3-tk
  Windows: Usually included with Python

Usage:
    python3 settings_gui.py

Author: Enhanced for Pharmacy Verification System
Version: 2.0
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pyautogui
from PIL import Image, ImageTk
import json
import os
import time
import shutil
from typing import Dict, Any, Optional, Tuple

class SettingsGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Pharmacy Verification - Settings & Coordinate Adjuster v2.0")
        self.root.geometry("1700x900")
        self.root.minsize(1500, 800)
        
        # Initialize all variables first
        self.config: Optional[Dict[str, Any]] = None
        self.current_field: Optional[str] = None
        self.current_region_type: Optional[str] = None
        
        # Screenshot and display
        self.screenshot: Optional[Image.Image] = None
        self.display_image: Optional[Image.Image] = None
        self.canvas: Optional[tk.Canvas] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.rectangles: Dict[str, int] = {}
        
        # UI state
        self.drag_start: Optional[Tuple[int, int]] = None
        self.current_rect: Optional[int] = None
        self.scale_x: float = 1.0
        self.scale_y: float = 1.0
        self.zoom_factor: float = 1.0
        
        # Pan functionality
        self.pan_start: Optional[Tuple[int, int]] = None
        self.canvas_offset_x: int = 0
        self.canvas_offset_y: int = 0
        self.is_panning: bool = False
        
        # Enhancement features
        self.show_labels = tk.BooleanVar(value=True)
        self.auto_save = tk.BooleanVar(value=False)
        self.preview_mode = tk.BooleanVar(value=False)
        self.optional_field_vars = {}
        
        # Initialize UI variables
        self.field_var: tk.StringVar = tk.StringVar()
        self.region_var: tk.StringVar = tk.StringVar(value="entered")
        self.coord_vars: Dict[str, tk.StringVar] = {}
        
        # UI components that will be created
        self.field_combo: Optional[ttk.Combobox] = None
        self.status_label: Optional[ttk.Label] = None
        self.canvas_frame: Optional[ttk.Frame] = None
        self.scrollable_frame: Optional[ttk.Frame] = None
        self.scroll_canvas: Optional[tk.Canvas] = None
        self.canvas_window: Optional[int] = None
        self.zoom_label: Optional[ttk.Label] = None
        
        # Load config and setup UI
        if not self.load_config():
            messagebox.showerror("Error", "Could not load config.json. Please ensure the file exists.")
            self.root.destroy()
            return
            
        self.setup_ui()
        self.take_screenshot()

    def load_config(self) -> bool:
        """Load configuration from config.json file."""
        try:
            config_path = "config.json"
            if not os.path.exists(config_path):
                # Try in same directory as script
                config_path = os.path.join(os.path.dirname(__file__), "config.json")
            
            with open(config_path, 'r') as f:
                self.config = json.load(f)
                
            # Validate config structure
            if not isinstance(self.config, dict):
                raise ValueError("Config must be a dictionary")
            if "regions" not in self.config:
                raise ValueError("Config missing 'regions' section")
            if "fields" not in self.config["regions"]:
                raise ValueError("Config missing 'regions.fields' section")
                
            return True
        except Exception as e:
            print(f"Error loading config: {e}")
            return False
    
    def save_config(self):
        """Save current configuration to config.json file."""
        if not self.config:
            messagebox.showerror("Error", "No configuration to save")
            return
            
        try:
            config_path = "config.json"
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            messagebox.showinfo("Success", "Configuration saved successfully!")
            self.update_status("Configuration saved")
            
            # Auto-save notification
            if self.auto_save.get():
                self.update_status("Auto-save: Configuration updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save config: {e}")
            self.update_status("Error saving configuration")
    
    def import_config(self):
        """Import configuration from a different file."""
        file_path = filedialog.askopenfilename(
            title="Import Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    new_config = json.load(f)
                self.config = new_config
                self.update_field_options()
                self.draw_all_rectangles()
                messagebox.showinfo("Success", "Configuration imported successfully!")
                self.update_status(f"Configuration imported from {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not import config: {e}")
                self.update_status("Error importing configuration")
    
    def export_config(self):
        """Export configuration to a different file."""
        if not self.config:
            messagebox.showerror("Error", "No configuration to export")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Export Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
                messagebox.showinfo("Success", f"Configuration exported to {file_path}")
                self.update_status(f"Configuration exported to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not export config: {e}")
                self.update_status("Error exporting configuration")
    
    def setup_ui(self):
        """Set up the complete user interface with enhanced features."""
        # Create menu bar
        self.create_menu()
        
        # Main frame with better layout
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Scrollable Controls (Wider to fit all content)
        left_container = ttk.Frame(main_frame, width=550)
        left_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_container.pack_propagate(False)
        
        # Create scrollable area for controls with proper frame structure
        # Main container for canvas and scrollbar
        scroll_container = ttk.Frame(left_container)
        scroll_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for scrolling
        self.scroll_canvas = tk.Canvas(scroll_container, highlightthickness=0, bg='SystemButtonFace')
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=self.scroll_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.scroll_canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))
        )
        
        # Create window in canvas
        self.canvas_window = self.scroll_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            self.scroll_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.scroll_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            self.scroll_canvas.unbind_all("<MouseWheel>")
        
        # Bind mouse enter/leave for scroll wheel
        self.scroll_canvas.bind('<Enter>', _bind_to_mousewheel)
        self.scroll_canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Configure canvas window width to match canvas width
        def _configure_canvas_window(event):
            canvas_width = event.width
            self.scroll_canvas.itemconfig(self.canvas_window, width=canvas_width-25)  # Account for scrollbar and wider content
        
        self.scroll_canvas.bind('<Configure>', _configure_canvas_window)
        
        self.setup_control_panel(self.scrollable_frame)
        
        # Right panel - Responsive Screenshot with zoom controls
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Zoom controls at top
        zoom_frame = ttk.Frame(right_frame)
        zoom_frame.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Label(zoom_frame, text="Zoom:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(zoom_frame, text="-", width=3, 
                  command=lambda: self.zoom(0.8)).pack(side=tk.LEFT, padx=2)
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(zoom_frame, text="+", width=3, 
                  command=lambda: self.zoom(1.25)).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(zoom_frame, text="Reset", width=6, 
                  command=lambda: self.zoom(1.0, reset=True)).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(zoom_frame, text="Fit to Window", width=12, 
                  command=self.fit_to_window).pack(side=tk.LEFT, padx=10)
        
        # Instructions for panning
        ttk.Label(zoom_frame, text="Hold Ctrl+Click to pan when zoomed", 
                 font=("Arial", 8, "italic")).pack(side=tk.RIGHT)
        
        # Canvas frame - now responsive
        self.canvas_frame = ttk.Frame(right_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Bind resize event to update canvas size
        self.canvas_frame.bind("<Configure>", self.on_canvas_frame_resize)
        
    def create_menu(self):
        """Create enhanced menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Config...", command=self.import_config)
        file_menu.add_command(label="Export Config...", command=self.export_config)
        file_menu.add_separator()
        file_menu.add_command(label="Save Config", command=self.save_config, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # View menu
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_checkbutton(label="Show Labels", variable=self.show_labels, command=self.draw_all_rectangles)
        view_menu.add_checkbutton(label="Preview Mode", variable=self.preview_mode, command=self.toggle_preview_mode)
        view_menu.add_separator()
        view_menu.add_command(label="Zoom In", command=lambda: self.zoom(1.2))
        view_menu.add_command(label="Zoom Out", command=lambda: self.zoom(0.8))
        view_menu.add_command(label="Reset Zoom", command=lambda: self.zoom(1.0, reset=True))
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Take New Screenshot", command=self.take_screenshot)
        tools_menu.add_command(label="Test OCR on Selection", command=self.test_ocr)
        tools_menu.add_checkbutton(label="Auto-Save", variable=self.auto_save)
        tools_menu.add_separator()
        tools_menu.add_command(label="Validate All Regions", command=self.validate_all_regions)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-s>', lambda e: self.save_config())
        self.root.bind('<F5>', lambda e: self.take_screenshot())
        
    def setup_control_panel(self, parent):
        """Set up the enhanced control panel."""
        # Header with better spacing
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Label(header_frame, text="Settings & Coordinate Adjuster v2.0", 
                 font=("Arial", 14, "bold")).pack()
        ttk.Label(header_frame, text="Enhanced Pharmacy Verification Tool", 
                 font=("Arial", 9, "italic")).pack(pady=(5, 0))
        
        # Instructions with better formatting and word wrap
        instr_frame = ttk.LabelFrame(parent, text="Instructions", padding=10)
        instr_frame.pack(fill=tk.X, pady=(0, 15))
        
        instructions = [
            "1. Select a field and region type",
            "2. Click and drag on the screenshot to define area",
            "3. Fine-tune coordinates manually if needed",
            "4. Adjust general settings (thresholds, automation)",
            "5. Use Ctrl+S to save, F5 for new screenshot",
            "6. Test OCR to verify selection quality", 
            "7. Select the smallest area possible for the best accuracy",
            "8. Avoid including any shapes, outlines, boxes, or other artifacts"
        ]
        
        for instr in instructions:
            # Create a label with word wrapping
            label = ttk.Label(instr_frame, text=instr, font=("Arial", 9), wraplength=520)
            label.pack(anchor=tk.W, pady=1, fill=tk.X)
        
        # Field and region selection
        select_frame = ttk.LabelFrame(parent, text="Selection", padding=10)
        select_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Field selection with validation
        ttk.Label(select_frame, text="Field:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        self.field_combo = ttk.Combobox(select_frame, textvariable=self.field_var, width=55, state="readonly")
        self.update_field_options()
        self.field_combo.pack(pady=(0, 10), fill=tk.X)
        self.field_combo.bind('<<ComboboxSelected>>', self.on_field_change)
        
        # Region type selection with enhanced layout
        ttk.Label(select_frame, text="Region Type:", font=("Arial", 10, "bold")).pack(anchor=tk.W)
        region_frame = ttk.Frame(select_frame)
        region_frame.pack(anchor=tk.W, pady=(0, 10))
        
        self.radio_entered = ttk.Radiobutton(region_frame, text="Entered (Left Panel)", 
                                           variable=self.region_var, value="entered", 
                                           command=self.on_region_change)
        self.radio_source = ttk.Radiobutton(region_frame, text="Source (Right Panel)", 
                                          variable=self.region_var, value="source", 
                                          command=self.on_region_change)
        self.radio_trigger = ttk.Radiobutton(region_frame, text="Trigger Detection", 
                                           variable=self.region_var, value="trigger", 
                                           command=self.on_region_change)
        
        self.radio_entered.pack(anchor=tk.W)
        self.radio_source.pack(anchor=tk.W)
        self.radio_trigger.pack(anchor=tk.W)
        
        # Enhanced coordinate controls with better layout
        coord_frame = ttk.LabelFrame(parent, text="Coordinates", padding=10)
        coord_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create a grid layout for coordinates
        coord_labels = ["X1 (Left):", "Y1 (Top):", "X2 (Right):", "Y2 (Bottom):"]
        
        # Create two columns for better space utilization
        for i in range(0, len(coord_labels), 2):
            row_frame = ttk.Frame(coord_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            # Left column
            left_col = ttk.Frame(row_frame)
            left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))
            
            ttk.Label(left_col, text=coord_labels[i], width=10).pack(side=tk.LEFT)
            var1 = tk.StringVar()
            entry1 = ttk.Entry(left_col, textvariable=var1, width=10)
            entry1.pack(side=tk.LEFT, padx=(5, 0))
            entry1.bind('<Return>', self.update_from_entries)
            entry1.bind('<FocusOut>', self.update_from_entries)
            self.coord_vars[f"coord_{i}"] = var1
            
            # Right column (if exists)
            if i + 1 < len(coord_labels):
                right_col = ttk.Frame(row_frame)
                right_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(8, 0))
                
                ttk.Label(right_col, text=coord_labels[i + 1], width=10).pack(side=tk.LEFT)
                var2 = tk.StringVar()
                entry2 = ttk.Entry(right_col, textvariable=var2, width=10)
                entry2.pack(side=tk.LEFT, padx=(5, 0))
                entry2.bind('<Return>', self.update_from_entries)
                entry2.bind('<FocusOut>', self.update_from_entries)
                self.coord_vars[f"coord_{i + 1}"] = var2
        
        # Enhanced button panel
        button_frame = ttk.LabelFrame(parent, text="Actions", padding=10)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Primary actions
        ttk.Button(button_frame, text="Update Rectangle", 
                  command=self.update_from_entries).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Reset to Config", 
                  command=self.reset_to_config).pack(fill=tk.X, pady=2)
        
        # Secondary actions
        ttk.Separator(button_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Take New Screenshot", 
                  command=self.take_screenshot).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Test OCR on Selection", 
                  command=self.test_ocr).pack(fill=tk.X, pady=2)
        ttk.Button(button_frame, text="Validate All Regions", 
                  command=self.validate_all_regions).pack(fill=tk.X, pady=2)
        
        # Save action (prominent)
        ttk.Separator(button_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        save_btn = ttk.Button(button_frame, text="💾 Save Configuration", 
                             command=self.save_config)
        save_btn.pack(fill=tk.X, pady=5)
        
        # General Settings Panel
        self.setup_settings_panel(parent)
        
        # Status display
        status_frame = ttk.LabelFrame(parent, text="Status", padding=10)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.status_label = ttk.Label(status_frame, text="Ready", 
                                     font=("Arial", 9))
        self.status_label.pack(anchor=tk.W)
    
    def setup_settings_panel(self, parent):
        """Set up the general settings panel."""
        settings_frame = ttk.LabelFrame(parent, text="General Settings", padding=10)
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Initialize settings variables if they don't exist
        if not hasattr(self, 'settings_vars'):
            self.settings_vars = {}
        
        # Threshold Settings
        threshold_frame = ttk.Frame(settings_frame)
        threshold_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(threshold_frame, text="Matching Thresholds (%):", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W)
        
        # Create threshold entries for each field type with better layout
        threshold_fields = [
            ("patient", "Patient Name"),
            ("prescriber", "Prescriber Name"), 
            ("drug", "Drug Name"),
            ("sig", "Directions/Sig"),
            ("patient_dob", "Patient DOB"),
            ("patient_address", "Patient Address"),
            ("patient_phone", "Patient Phone"),
            ("prescriber_address", "Prescriber Address")
        ]
        
        # Create two columns for threshold settings
        for i in range(0, len(threshold_fields), 2):
            row_frame = ttk.Frame(threshold_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            # Left column
            left_col = ttk.Frame(row_frame)
            left_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
            
            key1, label1 = threshold_fields[i]
            ttk.Label(left_col, text=f"{label1}:", width=14).pack(side=tk.LEFT)
            var1 = tk.StringVar()
            
            # Load current value from config
            if self.config and "thresholds" in self.config:
                current_value = self.config["thresholds"].get(key1, 65)
            else:
                current_value = 65
            var1.set(str(current_value))
            
            entry1 = ttk.Entry(left_col, textvariable=var1, width=6)
            entry1.pack(side=tk.LEFT, padx=(5, 0))
            entry1.bind('<FocusOut>', lambda e, k=key1: self.update_threshold(k))
            entry1.bind('<Return>', lambda e, k=key1: self.update_threshold(k))
            
            self.settings_vars[f"threshold_{key1}"] = var1
            ttk.Label(left_col, text="%").pack(side=tk.LEFT, padx=(2, 0))
            
            # Right column (if exists)
            if i + 1 < len(threshold_fields):
                right_col = ttk.Frame(row_frame)
                right_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                key2, label2 = threshold_fields[i + 1]
                ttk.Label(right_col, text=f"{label2}:", width=14).pack(side=tk.LEFT)
                var2 = tk.StringVar()
                
                # Load current value from config
                if self.config and "thresholds" in self.config:
                    current_value = self.config["thresholds"].get(key2, 65)
                else:
                    current_value = 65
                var2.set(str(current_value))
                
                entry2 = ttk.Entry(right_col, textvariable=var2, width=6)
                entry2.pack(side=tk.LEFT, padx=(5, 0))
                entry2.bind('<FocusOut>', lambda e, k=key2: self.update_threshold(k))
                entry2.bind('<Return>', lambda e, k=key2: self.update_threshold(k))
                
                self.settings_vars[f"threshold_{key2}"] = var2
                ttk.Label(right_col, text="%").pack(side=tk.LEFT, padx=(2, 0))
        
        # Automation Settings
        ttk.Separator(settings_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        auto_frame = ttk.Frame(settings_frame)
        auto_frame.pack(fill=tk.X)
        
        ttk.Label(auto_frame, text="Automation:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Enable/disable automation
        auto_enabled_var = tk.BooleanVar()
        if self.config and "automation" in self.config:
            current_auto = self.config["automation"].get("send_key_on_all_match", False)
        else:
            current_auto = False
        auto_enabled_var.set(current_auto)
        
        auto_check = ttk.Checkbutton(auto_frame, 
                                   text="Send key when all fields match",
                                   variable=auto_enabled_var,
                                   command=lambda: self.update_automation("enabled", auto_enabled_var.get()))
        auto_check.pack(anchor=tk.W, pady=2)
        self.settings_vars["auto_enabled"] = auto_enabled_var
        
        # Key selection and delay in one row
        controls_frame = ttk.Frame(auto_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        # Key selection
        key_frame = ttk.Frame(controls_frame)
        key_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
        
        ttk.Label(key_frame, text="Key to send:", width=12).pack(side=tk.LEFT)
        key_var = tk.StringVar()
        if self.config and "automation" in self.config:
            current_key = self.config["automation"].get("key_on_all_match", "f12")
        else:
            current_key = "f12"
        key_var.set(current_key)
        
        key_combo = ttk.Combobox(key_frame, textvariable=key_var, width=8, 
                               values=["f1", "f2", "f3", "f4", "f5", "f6", 
                                      "f7", "f8", "f9", "f10", "f11", "f12",
                                      "enter", "tab", "space", "escape"])
        key_combo.pack(side=tk.LEFT, padx=(5, 0))
        key_combo.bind('<<ComboboxSelected>>', lambda e: self.update_automation("key", key_var.get()))
        key_combo.bind('<FocusOut>', lambda e: self.update_automation("key", key_var.get()))
        
        self.settings_vars["auto_key"] = key_var
        
        # Key delay
        delay_frame = ttk.Frame(controls_frame)
        delay_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Label(delay_frame, text="Delay (sec):", width=12).pack(side=tk.LEFT)
        delay_var = tk.StringVar()
        if self.config and "automation" in self.config:
            current_delay = self.config["automation"].get("key_delay_seconds", 0.5)
        else:
            current_delay = 0.5
        delay_var.set(str(current_delay))
        
        delay_entry = ttk.Entry(delay_frame, textvariable=delay_var, width=6)
        delay_entry.pack(side=tk.LEFT, padx=(5, 0))
        delay_entry.bind('<FocusOut>', lambda e: self.update_automation("delay", delay_var.get()))
        delay_entry.bind('<Return>', lambda e: self.update_automation("delay", delay_var.get()))
        
        self.settings_vars["auto_delay"] = delay_var

        # Optional Fields Settings
        ttk.Separator(settings_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        optional_fields_frame = ttk.Frame(settings_frame)
        optional_fields_frame.pack(fill=tk.X)
        
        ttk.Label(optional_fields_frame, text="Optional Fields to Verify:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))

        if not hasattr(self, 'optional_field_vars'):
            self.optional_field_vars = {}
            
        self.optional_fields_config = {
            "patient_dob": "Patient DOB",
            "patient_address": "Patient Address",
            "patient_phone": "Patient Phone",
            "prescriber_address": "Prescriber Address"
        }
        
        enabled_fields = self.config.get("optional_fields_enabled", {})

        for key, label in self.optional_fields_config.items():
            var = tk.BooleanVar()
            var.set(enabled_fields.get(key, False))
            
            check = ttk.Checkbutton(optional_fields_frame, text=label, variable=var,
                                  command=lambda k=key: self.toggle_optional_field(k))
            check.pack(anchor=tk.W)
            self.optional_field_vars[key] = var
    
    def update_threshold(self, field_key):
        """Update threshold value in config."""
        if not self.config:
            return
            
        var_key = f"threshold_{field_key}"
        if var_key not in self.settings_vars:
            return
            
        try:
            new_value = int(self.settings_vars[var_key].get())
            if 0 <= new_value <= 100:
                if "thresholds" not in self.config:
                    self.config["thresholds"] = {}
                self.config["thresholds"][field_key] = new_value
                self.update_status(f"Updated {field_key} threshold to {new_value}%")
            else:
                raise ValueError("Threshold must be between 0 and 100")
        except ValueError as e:
            messagebox.showerror("Invalid Value", f"Threshold must be a number between 0 and 100")
            # Reset to previous value
            current_value = self.config.get("thresholds", {}).get(field_key, 65)
            self.settings_vars[var_key].set(str(current_value))
    
    def update_automation(self, setting_type, value):
        """Update automation settings in config."""
        if not self.config:
            return
            
        if "automation" not in self.config:
            self.config["automation"] = {}
            
        if setting_type == "enabled":
            self.config["automation"]["send_key_on_all_match"] = bool(value)
            self.update_status(f"Automation {'enabled' if value else 'disabled'}")
        elif setting_type == "key":
            self.config["automation"]["key_on_all_match"] = str(value)
            self.update_status(f"Automation key set to {value}")
        elif setting_type == "delay":
            try:
                delay_value = float(value)
                if delay_value >= 0:
                    self.config["automation"]["key_delay_seconds"] = delay_value
                    self.update_status(f"Key delay set to {delay_value}s")
                else:
                    raise ValueError("Delay must be non-negative")
            except ValueError:
                messagebox.showerror("Invalid Value", "Delay must be a non-negative number")
                # Reset to previous value
                current_delay = self.config.get("automation", {}).get("key_delay_seconds", 0.5)
                self.settings_vars["auto_delay"].set(str(current_delay))
    
    def toggle_optional_field(self, field_key):
        """Handle toggling of an optional field."""
        if not self.config:
            return

        if "optional_fields_enabled" not in self.config:
            self.config["optional_fields_enabled"] = {}
        
        is_enabled = self.optional_field_vars[field_key].get()
        self.config["optional_fields_enabled"][field_key] = is_enabled

        if is_enabled:
            # If enabled, ensure a placeholder exists in regions.fields
            if field_key not in self.config["regions"]["fields"]:
                threshold_key = field_key
                self.config["regions"]["fields"][field_key] = {
                    "entered": [0, 0, 0, 0],
                    "source": [0, 0, 0, 0],
                    "score_fn": "ratio",
                    "threshold_key": threshold_key
                }
                # Also need a threshold for it.
                if "thresholds" not in self.config:
                    self.config["thresholds"] = {}
                if threshold_key not in self.config["thresholds"]:
                    self.config["thresholds"][threshold_key] = 65 # default
        
        self.update_field_options()
        self.update_status(f"Optional field {field_key} {'enabled' if is_enabled else 'disabled'}")
        
        if self.auto_save.get():
            self.save_config()

    def update_field_options(self):
        """Update field options in the combobox."""
        if not self.config or not self.field_combo:
            return
            
        mandatory_fields = ["patient_name", "prescriber_name", "drug_name", "direction_sig"]
        
        enabled_optional_fields = []
        if "optional_fields_enabled" in self.config:
            for field, is_enabled in self.config["optional_fields_enabled"].items():
                if is_enabled:
                    enabled_optional_fields.append(field)

        field_options = ["trigger"] + mandatory_fields + enabled_optional_fields
        self.field_combo['values'] = sorted(list(set(field_options)))
    
    def update_status(self, message: str):
        """Update the status label with a message."""
        if self.status_label:
            self.status_label.config(text=message)
            self.root.after(3000, lambda: self.status_label.config(text="Ready"))
    
    def toggle_preview_mode(self):
        """Toggle preview mode for better viewing."""
        if self.preview_mode.get():
            self.update_status("Preview mode enabled")
            # In preview mode, make rectangles semi-transparent
            self.draw_all_rectangles()
        else:
            self.update_status("Preview mode disabled")
            self.draw_all_rectangles()
    
    def zoom(self, factor: float, reset: bool = False):
        """Zoom in or out of the screenshot."""
        if reset:
            self.zoom_factor = 1.0
        else:
            self.zoom_factor *= factor
            
        # Limit zoom range
        self.zoom_factor = max(0.25, min(5.0, self.zoom_factor))
        
        # Update zoom label
        if hasattr(self, 'zoom_label'):
            self.zoom_label.config(text=f"{int(self.zoom_factor * 100)}%")
        
        # Update display
        if self.screenshot:
            self._update_display_image()
            self.draw_all_rectangles()
        
        self.update_status(f"Zoom: {self.zoom_factor:.1f}x")
    
    def fit_to_window(self):
        """Fit the screenshot to the current window size."""
        if not self.screenshot or not self.canvas_frame:
            return
            
        # Get available space
        self.canvas_frame.update_idletasks()
        available_width = self.canvas_frame.winfo_width() - 20  # Padding
        available_height = self.canvas_frame.winfo_height() - 20  # Padding
        
        if available_width <= 0 or available_height <= 0:
            return
            
        # Calculate zoom factor to fit
        width_ratio = available_width / self.screenshot.width
        height_ratio = available_height / self.screenshot.height
        
        # Use the smaller ratio to ensure it fits both dimensions
        self.zoom_factor = min(width_ratio, height_ratio)
        self.zoom_factor = max(0.1, min(5.0, self.zoom_factor))  # Limit range
        
        # Reset pan offset
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
        
        # Update display
        self._update_display_image()
        self.draw_all_rectangles()
        
        # Update zoom label
        if hasattr(self, 'zoom_label'):
            self.zoom_label.config(text=f"{int(self.zoom_factor * 100)}%")
        
        self.update_status("Fitted to window")
    
    def on_canvas_frame_resize(self, event):
        """Handle canvas frame resize to update screenshot display."""
        if self.screenshot and hasattr(self, 'canvas') and self.canvas:
            # Delay the update slightly to avoid too many updates during resize
            self.root.after(100, self._delayed_resize_update)
    
    def _delayed_resize_update(self):
        """Delayed update after resize to avoid flickering."""
        if self.screenshot:
            self._update_display_image()
            self.draw_all_rectangles()
    
    def _update_display_image(self):
        """Update the display image with current zoom factor and responsive sizing."""
        if not self.screenshot or not self.canvas_frame:
            return
            
        # Get available space
        self.canvas_frame.update_idletasks()
        available_width = self.canvas_frame.winfo_width()
        available_height = self.canvas_frame.winfo_height()
        
        if available_width <= 1 or available_height <= 1:
            # Frame not ready yet
            self.root.after(50, self._update_display_image)
            return
        
        # Calculate display size based on zoom and available space
        display_width = int(self.screenshot.width * self.zoom_factor)
        display_height = int(self.screenshot.height * self.zoom_factor)
        
        # Create resized image
        self.display_image = self.screenshot.resize(
            (display_width, display_height), 
            Image.Resampling.LANCZOS
        )
        self.photo = ImageTk.PhotoImage(self.display_image)
        
        # Clear the canvas frame completely
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()
        
        # Create new container for canvas and scrollbars
        canvas_container = ttk.Frame(self.canvas_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Calculate canvas size (limit to available space for scrolling)
        canvas_width = min(display_width, available_width - 20)
        canvas_height = min(display_height, available_height - 20)
        
        # Create new canvas
        self.canvas = tk.Canvas(
            canvas_container, 
            width=canvas_width, 
            height=canvas_height,
            scrollregion=(0, 0, display_width, display_height)
        )
        
        # Add scrollbars only if needed
        need_h_scroll = display_width > canvas_width
        need_v_scroll = display_height > canvas_height
        
        if need_v_scroll:
            v_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
            v_scrollbar.pack(side="right", fill="y")
            self.canvas.configure(yscrollcommand=v_scrollbar.set)
            
        if need_h_scroll:
            h_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.canvas.xview)
            h_scrollbar.pack(side="bottom", fill="x")
            self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Pack canvas
        self.canvas.pack(side="left", fill="both", expand=True)
        
        # Create image on canvas
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Bind panning events (Ctrl+drag)
        self.canvas.bind("<Control-Button-1>", self.on_pan_start)
        self.canvas.bind("<Control-B1-Motion>", self.on_pan_drag)
        self.canvas.bind("<Control-ButtonRelease-1>", self.on_pan_end)
        
        # Bind mouse wheel for zooming
        self.canvas.bind("<Control-MouseWheel>", self.on_wheel_zoom)
        
        # Update scale factors for coordinate conversion
        self.scale_x = self.screenshot.width / display_width
        self.scale_y = self.screenshot.height / display_height
        
        # Reset canvas offset since we're creating a new canvas
        self.canvas_offset_x = 0
        self.canvas_offset_y = 0
    
    def on_pan_start(self, event):
        """Start panning the image."""
        self.pan_start = (event.x, event.y)
        self.is_panning = True
        if self.canvas:
            self.canvas.config(cursor="fleur")
    
    def on_pan_drag(self, event):
        """Handle panning drag."""
        if self.pan_start and self.is_panning and self.canvas:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            
            # Update canvas view using built-in scrolling
            self.canvas.scan_mark(self.pan_start[0], self.pan_start[1])
            self.canvas.scan_dragto(event.x, event.y, gain=1)
    
    def on_pan_end(self, event):
        """End panning."""
        self.is_panning = False
        self.pan_start = None
        if self.canvas:
            self.canvas.config(cursor="")
    
    def on_wheel_zoom(self, event):
        """Handle mouse wheel zoom."""
        if event.delta > 0:
            self.zoom(1.1)
        else:
            self.zoom(0.9)
    
    def test_ocr(self):
        """Test OCR on the currently selected region."""
        if not self.current_field or not self.current_region_type:
            messagebox.showwarning("Warning", "Please select a field and region type first!")
            return
            
        try:
            # Get coordinates
            if self.current_field == "trigger":
                coords = self.config["regions"]["trigger"]
            else:
                coords = self.config["regions"]["fields"][self.current_field][self.current_region_type]
            
            # Take screenshot and crop
            screenshot = pyautogui.screenshot()
            cropped = screenshot.crop(coords)
            
            # Try OCR
            import pytesseract
            text = pytesseract.image_to_string(cropped).strip()
            
            # Show result
            if text:
                messagebox.showinfo("OCR Test Result", f"Detected text:\n\n{text}")
                self.update_status("OCR test completed successfully")
            else:
                messagebox.showwarning("OCR Test Result", "No text detected in the selected region.")
                self.update_status("OCR test: No text detected")
                
        except ImportError:
            messagebox.showerror("Error", "pytesseract is not installed. Install it with: pip install pytesseract")
        except Exception as e:
            messagebox.showerror("Error", f"OCR test failed: {e}")
            self.update_status("OCR test failed")
    
    def validate_all_regions(self):
        """Validate all defined regions for potential issues."""
        if not self.config:
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
            messagebox.showwarning("Validation Issues", "Found issues:\n\n" + "\n".join(issues))
            self.update_status(f"Validation: {len(issues)} issues found")
        else:
            messagebox.showinfo("Validation Complete", "All regions are valid!")
            self.update_status("Validation: All regions valid")
        
    def take_screenshot(self):
        """Take a new screenshot for region adjustment."""
        try:
            # Hide window temporarily
            self.root.withdraw()
            self.update_status("Taking screenshot...")
            self.root.after(500, self._take_screenshot_delayed)
        except Exception as e:
            messagebox.showerror("Error", f"Could not take screenshot: {e}")
            self.root.deiconify()
    
    def _take_screenshot_delayed(self):
        """Take screenshot after delay to hide window."""
        try:
            self.screenshot = pyautogui.screenshot()
            
            # Reset pan offset for new screenshot
            self.canvas_offset_x = 0
            self.canvas_offset_y = 0
            
            # Fit to window by default for new screenshot
            self.fit_to_window()
            
            self.drag_start = None
            self.current_rect = None
            
            self.root.deiconify()
            self.draw_all_rectangles()
            self.update_status("Screenshot updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not process screenshot: {e}")
            self.root.deiconify()
            self.update_status("Screenshot failed")
    
    def on_click(self, event):
        """Handle mouse click on canvas."""
        if not self.current_field or not self.current_region_type:
            messagebox.showwarning("Warning", "Please select a field and region type first!")
            return
            
        # Don't start rectangle drawing if we're panning
        if self.is_panning:
            return
            
        # Get canvas coordinates accounting for scroll position
        if self.canvas:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
        else:
            canvas_x = event.x
            canvas_y = event.y
        
        self.drag_start = (canvas_x, canvas_y)
        if self.current_rect and self.canvas:
            self.canvas.delete(self.current_rect)
    
    def on_drag(self, event):
        """Handle mouse drag on canvas."""
        if self.drag_start and self.canvas and not self.is_panning:
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            
            # Get canvas coordinates accounting for scroll position
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            
            x1, y1 = self.drag_start
            x2, y2 = canvas_x, canvas_y
            
            # Ensure x1,y1 is top-left and x2,y2 is bottom-right
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            self.current_rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
    
    def on_release(self, event):
        """Handle mouse release on canvas."""
        if self.drag_start and self.current_field and self.current_region_type and self.config and not self.is_panning:
            # Get canvas coordinates accounting for scroll position
            if self.canvas:
                canvas_x = self.canvas.canvasx(event.x)
                canvas_y = self.canvas.canvasy(event.y)
            else:
                canvas_x = event.x
                canvas_y = event.y
            
            x1, y1 = self.drag_start
            x2, y2 = canvas_x, canvas_y
            
            # Ensure correct order
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Convert to actual screenshot coordinates (no offset needed since canvas shows image directly)
            actual_x1 = int(x1 * self.scale_x)
            actual_y1 = int(y1 * self.scale_y)
            actual_x2 = int(x2 * self.scale_x)
            actual_y2 = int(y2 * self.scale_y)
            
            # Ensure coordinates are within screenshot bounds
            if self.screenshot:
                actual_x1 = max(0, min(actual_x1, self.screenshot.width))
                actual_y1 = max(0, min(actual_y1, self.screenshot.height))
                actual_x2 = max(0, min(actual_x2, self.screenshot.width))
                actual_y2 = max(0, min(actual_y2, self.screenshot.height))
            
            # Update config
            if self.current_field == "trigger":
                self.config["regions"]["trigger"] = [actual_x1, actual_y1, actual_x2, actual_y2]
            else:
                self.config["regions"]["fields"][self.current_field][self.current_region_type] = [
                    actual_x1, actual_y1, actual_x2, actual_y2
                ]
            
            # Update coordinate entries
            self.coord_vars["coord_0"].set(str(actual_x1))
            self.coord_vars["coord_1"].set(str(actual_y1))
            self.coord_vars["coord_2"].set(str(actual_x2))
            self.coord_vars["coord_3"].set(str(actual_y2))
            
            # Store rectangle
            rect_key = f"{self.current_field}_{self.current_region_type}"
            if self.current_rect:
                self.rectangles[rect_key] = self.current_rect
            
            self.draw_all_rectangles()
            self.update_status(f"Updated {self.current_field} {self.current_region_type}")
            
            # Auto-save if enabled
            if self.auto_save.get():
                self.save_config()
    
    def on_field_change(self, event=None):
        """Handle field selection change."""
        self.current_field = self.field_var.get()
        
        # Show/hide region type options based on field selection
        if self.current_field == "trigger":
            self.radio_entered.pack_forget()
            self.radio_source.pack_forget()
            self.radio_trigger.pack(anchor=tk.W)
            self.region_var.set("trigger")
            self.current_region_type = "trigger"
        else:
            self.radio_trigger.pack_forget()
            self.radio_entered.pack(anchor=tk.W)
            self.radio_source.pack(anchor=tk.W)
            if self.region_var.get() == "trigger":
                self.region_var.set("entered")
            self.current_region_type = self.region_var.get()
            
        self.update_coordinate_display()
        self.draw_all_rectangles()
        self.update_status(f"Selected field: {self.current_field}")
    
    def on_region_change(self):
        """Handle region type selection change."""
        self.current_region_type = self.region_var.get()
        self.update_coordinate_display()
        self.draw_all_rectangles()
        self.update_status(f"Selected region: {self.current_region_type}")
    
    def update_coordinate_display(self):
        """Update coordinate entry fields with current values."""
        if self.current_field and self.current_region_type and self.config:
            try:
                if self.current_field == "trigger":
                    coords = self.config["regions"]["trigger"]
                else:
                    coords = self.config["regions"]["fields"][self.current_field][self.current_region_type]
                
                for i, coord in enumerate(coords):
                    self.coord_vars[f"coord_{i}"].set(str(coord))
            except (KeyError, IndexError):
                # Clear coordinates if not found
                for i in range(4):
                    self.coord_vars[f"coord_{i}"].set("")
    
    def update_from_entries(self, event=None):
        """Update configuration from coordinate entry fields."""
        if not self.current_field or not self.current_region_type or not self.config:
            return
            
        try:
            coords = [int(self.coord_vars[f"coord_{i}"].get()) for i in range(4)]
            
            # Validate coordinates
            if coords[0] >= coords[2] or coords[1] >= coords[3]:
                messagebox.showerror("Error", "Invalid coordinates: x1 must be < x2 and y1 must be < y2")
                return
            
            if self.current_field == "trigger":
                self.config["regions"]["trigger"] = coords
            else:
                self.config["regions"]["fields"][self.current_field][self.current_region_type] = coords
            
            self.draw_all_rectangles()
            self.update_status("Coordinates updated manually")
            
            # Auto-save if enabled
            if self.auto_save.get():
                self.save_config()
                
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integer coordinates")
    
    def reset_to_config(self):
        """Reset coordinate display to configuration values."""
        self.update_coordinate_display()
        self.draw_all_rectangles()
        self.update_status("Reset to configuration values")
    
    def draw_all_rectangles(self):
        """Draw all configured rectangles on the canvas."""
        if not self.canvas or not self.config or not self.screenshot:
            return
            
        # Clear existing rectangles
        for rect in self.rectangles.values():
            if rect:
                try:
                    self.canvas.delete(rect)
                except:
                    pass  # Rectangle might not exist anymore
        self.rectangles.clear()
        
        # Draw trigger region
        if "trigger" in self.config["regions"]:
            coords = self.config["regions"]["trigger"]
            if len(coords) >= 4:
                # Convert from screenshot coordinates to display coordinates
                x1 = int(coords[0] / self.scale_x)
                y1 = int(coords[1] / self.scale_y)
                x2 = int(coords[2] / self.scale_x)
                y2 = int(coords[3] / self.scale_y)
                
                color = "red"
                width = 3 if (self.current_field == "trigger") else 1
                
                rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
                self.rectangles["trigger"] = rect
                
                # Add label if enabled
                if self.show_labels.get():
                    label_x = x1 + 5
                    label_y = y1 + 5
                    self.canvas.create_text(label_x, label_y, text="trigger", anchor=tk.NW, 
                                          fill=color, font=("Arial", 8, "bold"))
        
        # Draw all field rectangles
        colors = {"entered": "blue", "source": "green"}
        
        if "fields" in self.config["regions"]:
            for field_name, field_config in self.config["regions"]["fields"].items():
                for region_type in ["entered", "source"]:
                    if region_type in field_config and len(field_config[region_type]) >= 4:
                        coords = field_config[region_type]
                        
                        # Convert to display coordinates (no offset needed since canvas shows image directly)
                        x1 = int(coords[0] / self.scale_x)
                        y1 = int(coords[1] / self.scale_y)
                        x2 = int(coords[2] / self.scale_x)
                        y2 = int(coords[3] / self.scale_y)
                        
                        color = colors[region_type]
                        width = 3 if (field_name == self.current_field and region_type == self.current_region_type) else 1
                        
                        rect = self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
                        self.rectangles[f"{field_name}_{region_type}"] = rect
                        
                        # Add label if enabled
                        if self.show_labels.get():
                            label_x = x1 + 5
                            label_y = y1 + 5
                            text = f"{field_name}_{region_type}"
                            self.canvas.create_text(label_x, label_y, text=text, anchor=tk.NW, 
                                                  fill=color, font=("Arial", 8, "bold"))
    
    def run(self):
        """Start the application main loop."""
        self.root.mainloop()

if __name__ == "__main__":
    app = SettingsGUI()
    app.run()
