#!/usr/bin/env python3
"""
VLM Configuration GUI
====================

Visual interface for configuring VLM screenshot regions.
Allows users to take screenshots and select regions by clicking and dragging.

Usage:
    python vlm_gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import sys
from PIL import Image, ImageTk, ImageDraw
import pyautogui
from typing import Dict, List, Tuple, Optional
import threading
import time
import re
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def substitute_env_vars(config_dict: Dict) -> Dict:
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

class VLMRegionSelector:
    """GUI for selecting VLM screenshot regions"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VLM Configuration - Region Selector")
        self.root.geometry("1200x800")
        
        # Configuration
        self.vlm_config_file = os.path.join("config", "vlm_config.json")
        self.vlm_config = self.load_vlm_config()
        
        # Screenshot and selection variables
        self.screenshot = None
        self.screenshot_tk = None
        self.canvas = None
        self.scale_factor = 1.0
        
        # Selection state
        self.selecting = False
        self.start_x = 0
        self.start_y = 0
        self.current_rect = None
        self.current_region = None
        
        # Single region storage for comparison area
        self.regions = {
            "comparison": [0, 0, 1600, 600]
        }
        
        self.setup_ui()
        self.load_current_regions()
        
    def load_vlm_config(self) -> Dict:
        """Load VLM configuration from file with environment variable substitution"""
        try:
            if os.path.exists(self.vlm_config_file):
                with open(self.vlm_config_file, 'r') as f:
                    raw_config = json.load(f)
                
                # Substitute environment variables
                return substitute_env_vars(raw_config)
            else:
                return self.create_default_config()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load VLM config: {e}")
            return self.create_default_config()
    
    def create_default_config(self) -> Dict:
        """Create default VLM configuration"""
        return {
            "vlm_config": {
                "base_url": "http://localhost:8081/v1",
                "api_key": "llamacpp",
                "model_name": ".\\models\\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                "oneshot_system_prompt": "You are a pharmacy prescription verification agent. Compare the prescription data and provide accurate matching scores.",
                "oneshot_user_prompt": """You are a pharmacy prescription verify agent. In the screenshot, the left side is what was entered, the right side is the original prescription.

Please give me a matching score between 0-100 for:
- patient: How well does the patient information match?
- prescriber: How well does the prescriber information match?
- drug: How well does the drug information match?
- direction: How well do the directions match?

Return your response in this exact JSON format (no reasoning, no explanation):
{
  "patient": score,
  "prescriber": score,
  "drug": score,
  "direction": score
}""",
                "max_tokens": 500,
                "temperature": 0.0
            },
            "vlm_regions": {
                "comparison": [0, 0, 1600, 600]
            },
            "vlm_settings": {
                "image_format": "PNG",
                "image_quality": 95,
                "auto_enhance": True,
                "resize_max_width": 1920,
                "resize_max_height": 1440
            }
        }
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="VLM Region Configuration", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Left panel - Controls
        controls_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        controls_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Screenshot button
        ttk.Button(controls_frame, text="📸 Take Screenshot", 
                  command=self.take_screenshot, width=20).grid(row=0, column=0, pady=5, sticky=tk.W)
        
        # Region selection
        ttk.Label(controls_frame, text="Comparison Region:").grid(row=1, column=0, pady=(20, 5), sticky=tk.W)
        
        info_label = ttk.Label(controls_frame, 
                              text="Select the area containing BOTH\nthe data entry (left) and\noriginal prescription (right)", 
                              font=("Arial", 9),
                              foreground="blue")
        info_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 10))
        
        # Current coordinates display
        coords_frame = ttk.LabelFrame(controls_frame, text="Current Coordinates", padding="10")
        coords_frame.grid(row=3, column=0, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.coords_text = tk.Text(coords_frame, height=8, width=25)
        self.coords_text.grid(row=0, column=0)
        
        # Action buttons
        button_frame = ttk.Frame(controls_frame)
        button_frame.grid(row=4, column=0, pady=(20, 0), sticky=tk.W)
        
        ttk.Button(button_frame, text="💾 Save Configuration", 
                  command=self.save_config).grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="🔄 Load Configuration", 
                  command=self.load_current_regions).grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(button_frame, text="🧪 Test VLM", 
                  command=self.test_vlm).grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E))
        
        button_frame.columnconfigure(0, weight=1)
        
        # Right panel - Screenshot canvas
        canvas_frame = ttk.LabelFrame(main_frame, text="Screenshot - Click and drag to select regions", padding="10")
        canvas_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)
        
        # Canvas with scrollbars
        canvas_container = ttk.Frame(canvas_frame)
        canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(canvas_container, bg="white")
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Take a screenshot to begin")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def take_screenshot(self):
        """Take a screenshot and display it in the canvas"""
        try:
            self.status_var.set("Taking screenshot in 2 seconds... Get ready!")
            self.root.update()
            
            # Minimize window and wait
            self.root.withdraw()
            time.sleep(2)
            
            # Take screenshot
            self.screenshot = pyautogui.screenshot()
            
            # Restore window
            self.root.deiconify()
            
            # Calculate scale factor to fit canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 800
                canvas_height = 600
            
            img_width, img_height = self.screenshot.size
            
            scale_x = (canvas_width - 20) / img_width
            scale_y = (canvas_height - 20) / img_height
            self.scale_factor = min(scale_x, scale_y, 1.0)  # Don't scale up
            
            # Resize screenshot for display
            display_width = int(img_width * self.scale_factor)
            display_height = int(img_height * self.scale_factor)
            
            display_image = self.screenshot.resize((display_width, display_height), Image.Resampling.LANCZOS)
            self.screenshot_tk = ImageTk.PhotoImage(display_image)
            
            # Update canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.screenshot_tk)
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            
            # Draw existing regions
            self.draw_regions()
            
            self.status_var.set(f"Screenshot captured ({img_width}x{img_height}). Select a region type and click-drag to define coordinates.")
            
        except Exception as e:
            self.status_var.set(f"Error taking screenshot: {e}")
            messagebox.showerror("Error", f"Failed to take screenshot: {e}")
    
    def draw_regions(self):
        """Draw existing region on the canvas"""
        if not self.screenshot:
            return
        
        # Single region with green outline
        region_name = "comparison"
        coords = self.regions.get(region_name, [])
        
        if len(coords) == 4:
            x1, y1, x2, y2 = coords
            
            # Scale coordinates for display
            x1_scaled = x1 * self.scale_factor
            y1_scaled = y1 * self.scale_factor
            x2_scaled = x2 * self.scale_factor
            y2_scaled = y2 * self.scale_factor
            
            color = "green"
            
            # Draw rectangle
            self.canvas.create_rectangle(x1_scaled, y1_scaled, x2_scaled, y2_scaled,
                                       outline=color, width=3, tags=f"region_{region_name}")
            
            # Draw label
            label_x = x1_scaled + 5
            label_y = y1_scaled + 5
            self.canvas.create_text(label_x, label_y, text="Comparison Region",
                                  fill=color, font=("Arial", 12, "bold"), anchor=tk.NW,
                                  tags=f"region_{region_name}")
    
    def on_canvas_click(self, event):
        """Handle canvas click - start region selection"""
        if not self.screenshot:
            messagebox.showwarning("Warning", "Please take a screenshot first!")
            return
        
        self.selecting = True
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.current_region = "comparison"  # Always "comparison" now
        
        # Remove existing selection rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        self.status_var.set(f"Selecting comparison region...")
    
    def on_canvas_drag(self, event):
        """Handle canvas drag - update selection rectangle"""
        if not self.selecting:
            return
        
        current_x = self.canvas.canvasx(event.x)
        current_y = self.canvas.canvasy(event.y)
        
        # Remove previous rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        # Draw new rectangle (always green for comparison)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, current_x, current_y,
            outline="green", width=3, dash=(5, 5)
        )
    
    def on_canvas_release(self, event):
        """Handle canvas release - finalize region selection"""
        if not self.selecting:
            return
        
        end_x = self.canvas.canvasx(event.x)
        end_y = self.canvas.canvasy(event.y)
        
        # Convert back to screen coordinates
        x1 = int(min(self.start_x, end_x) / self.scale_factor)
        y1 = int(min(self.start_y, end_y) / self.scale_factor)
        x2 = int(max(self.start_x, end_x) / self.scale_factor)
        y2 = int(max(self.start_y, end_y) / self.scale_factor)
        
        # Validate selection
        if abs(x2 - x1) < 10 or abs(y2 - y1) < 10:
            messagebox.showwarning("Warning", "Selection too small! Please select a larger area.")
            if self.current_rect:
                self.canvas.delete(self.current_rect)
            self.selecting = False
            return
        
        # Store coordinates (always "comparison")
        self.regions["comparison"] = [x1, y1, x2, y2]
        
        # Remove temporary rectangle
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        
        # Redraw region
        self.canvas.delete("region_comparison")
        self.draw_regions()
        
        # Update coordinates display
        self.update_coordinates_display()
        
        self.selecting = False
        self.status_var.set(f"Comparison region updated: {x1}, {y1}, {x2}, {y2}")
    
    def update_coordinates_display(self):
        """Update the coordinates text display"""
        self.coords_text.delete(1.0, tk.END)
        
        coords_info = "Current Region:\n\n"
        
        region_name = "comparison"
        coords = self.regions.get(region_name, [])
        
        if coords:
            coords_info += f"Comparison Area:\n"
            coords_info += f"  X1: {coords[0]}\n"
            coords_info += f"  Y1: {coords[1]}\n"
            coords_info += f"  X2: {coords[2]}\n"
            coords_info += f"  Y2: {coords[3]}\n"
            coords_info += f"  Size: {coords[2]-coords[0]}×{coords[3]-coords[1]}\n\n"
            coords_info += f"This region should include BOTH\n"
            coords_info += f"the data entry screen (left) and\n"
            coords_info += f"the original prescription (right)."
        
        self.coords_text.insert(1.0, coords_info)
    
    def load_current_regions(self):
        """Load current regions from config file"""
        try:
            self.vlm_config = self.load_vlm_config()
            vlm_regions = self.vlm_config.get("vlm_regions", {})
            
            if vlm_regions:
                self.regions.update(vlm_regions)
                self.update_coordinates_display()
                
                if self.screenshot:
                    self.canvas.delete("region_comparison")
                    self.draw_regions()
                
                self.status_var.set("Regions loaded from configuration file")
            else:
                self.status_var.set("No regions found in configuration file")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load regions: {e}")
    
    def save_config(self):
        """Save current configuration to file"""
        try:
            self.vlm_config["vlm_regions"] = self.regions.copy()
            
            # Create backup in proper backup directory
            if os.path.exists(self.vlm_config_file):
                backup_dir = "config_backups"
                os.makedirs(backup_dir, exist_ok=True)
                
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_file = os.path.join(backup_dir, f"vlm_config_backup_{timestamp}.json")
                
                with open(self.vlm_config_file, 'r') as f:
                    backup_content = f.read()
                with open(backup_file, 'w') as f:
                    f.write(backup_content)
            
            # Save new configuration
            with open(self.vlm_config_file, 'w') as f:
                json.dump(self.vlm_config, f, indent=2)
            
            self.status_var.set("Configuration saved successfully!")
            messagebox.showinfo("Success", f"VLM configuration saved to {self.vlm_config_file}")
            
        except Exception as e:
            self.status_var.set(f"Error saving configuration: {e}")
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
    
    def test_vlm(self):
        """Test VLM with current configuration"""
        try:
            self.status_var.set("Testing VLM configuration...")
            
            # Run test in thread to avoid blocking UI
            def run_test():
                try:
                    # Test VLM connection using built-in test method
                    from ai.vlm_verifier import VLM_Verifier
                    vlm_verifier = VLM_Verifier(self.vlm_config)
                    result = vlm_verifier.test_vlm_connection()
                    
                    if "error" not in result:
                        self.root.after(0, lambda: self.show_test_result("Success", "VLM test passed! ✅"))
                    else:
                        self.root.after(0, lambda: self.show_test_result("Error", f"VLM test failed:\n{result.get('error', 'Unknown error')}"))
                        
                except Exception as e:
                    self.root.after(0, lambda: self.show_test_result("Error", f"Test error: {e}"))
            
            threading.Thread(target=run_test, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start test: {e}")
    
    def show_test_result(self, title, message):
        """Show test result in UI thread"""
        if title == "Success":
            messagebox.showinfo(title, message)
            self.status_var.set("VLM test completed successfully!")
        else:
            messagebox.showerror(title, message)
            self.status_var.set("VLM test failed - check configuration")
    
    def run(self):
        """Start the GUI application"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            print("\nApplication closed by user")

def main():
    """Main entry point"""
    print("🖥️  Starting VLM Configuration GUI...")
    
    try:
        app = VLMRegionSelector()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
