import streamlit as st
import json
import os
import sys
import time
from PIL import Image
import pyautogui
from typing import Dict, Any, Optional
import threading
import logging
import subprocess
import re
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def launch_vlm_gui():
    """Launch the VLM Configuration GUI"""
    try:
        # Launch the VLM GUI as a separate process
        # Get the directory where this script is located (ui folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vlm_gui_path = os.path.join(script_dir, "vlm_gui.py")
        
        if not os.path.exists(vlm_gui_path):
            st.error(f"❌ VLM GUI file not found: {vlm_gui_path}")
            return
        
        # Launch the GUI in a separate process
        if sys.platform.startswith('win'):
            # Windows
            subprocess.Popen([sys.executable, vlm_gui_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE)
        else:
            # macOS/Linux
            subprocess.Popen([sys.executable, vlm_gui_path])
        
        st.success("🚀 VLM Configuration GUI launched!")
        st.info("📋 A new window should open with the coordinate selection interface")
        
    except Exception as e:
        st.error(f"❌ Failed to launch VLM GUI: {e}")
        st.info("💡 Try running manually: python vlm_gui.py")

def vlm_settings_page(main_config: Dict[str, Any]):
    """VLM Settings and Configuration Page"""
    
    st.title("👁️ Vision Language Model Settings")
    st.info("🎯 **VLM Mode:** Use AI vision to directly compare prescription screenshots without OCR")
    
    st.warning("📋 **Setup Required:** This connects to your existing llamacpp server. Make sure your server is running with a vision model and `--multimodal` flag.")
    
    # Load VLM config
    vlm_config_file = os.path.join("config", "vlm_config.json")
    vlm_config = load_vlm_config(vlm_config_file)
    
    if not vlm_config:
        st.error("❌ Failed to load VLM configuration")
        return
    
    # Main tabs with prompt customization
    tab1, tab2, tab3 = st.tabs(["🔧 Model Settings", "📝 Prompt Customization", "🧪 Testing"])

    with tab1:
        show_model_settings(vlm_config, vlm_config_file)

    with tab2:
        show_prompt_customization(vlm_config, vlm_config_file)

    with tab3:
        show_vlm_testing(vlm_config)

def load_vlm_config(config_file: str) -> Optional[Dict[str, Any]]:
    """Load VLM configuration from file with environment variable substitution"""
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            # Substitute environment variables
            return substitute_env_vars(raw_config)
        else:
            # Create default config if it doesn't exist
            default_config = {
                "vlm_config": {
                    "base_url": "http://localhost:8080/v1",
                    "api_key": "llamacpp",
                    "model_name": "your-model-name",
                    "system_prompt": "You are a prescription verification assistant. Compare the entered prescription data (first image) with the source prescription (second image).\\n\\nAnalyze the following fields if visible:\\n- patient_name\\n- prescriber_name\\n- drug_name\\n- direction_sig (directions for use)\\n\\nFor each field, provide a confidence score from 0-100 based on how well the entered data matches the source:\\n- 90-100: Perfect or near-perfect match\\n- 70-89: Good match with minor differences\\n- 50-69: Moderate match with some discrepancies\\n- 30-49: Poor match with significant differences\\n- 0-29: Very poor match or completely different\\n\\nFormat your response as:\\nfield_name: score\\n\\nOnly include fields that are clearly visible in both images.",
                    "user_prompt": "Please compare these prescription images. First image shows entered data, second shows the source prescription. Analyze all visible prescription fields and provide confidence scores.",
                    "max_tokens": 500,
                    "temperature": 0.1
                },
                "vlm_regions": {
                    "data_entry": [0, 0, 800, 600],
                    "source": [800, 0, 1600, 600]
                },
                "vlm_settings": {
                    "image_format": "PNG",
                    "image_quality": 95,
                    "auto_enhance": True,
                    "resize_max_width": 1024,
                    "resize_max_height": 768,
                    "max_image_tokens_total": 1024,
                    "max_image_tokens_per_image": 512
                }
            }
            save_vlm_config(default_config, config_file)
            return default_config
    except Exception as e:
        st.error(f"Error loading VLM config: {e}")
        return None

def save_vlm_config(config: Dict[str, Any], config_file: str) -> bool:
    """Save VLM configuration to file"""
    try:
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"Error saving VLM config: {e}")
        return False

def show_prompt_customization(vlm_config: Dict[str, Any], config_file: str):
    """Show VLM prompt customization for one-shot mode"""
    st.subheader("📝 One-Shot VLM Prompt Customization")
    st.info("🎯 **One-Shot Mode:** Single image with both sides (left=entered, right=source) → Direct comparison and scoring")
    st.info("💡 **Shared Prompts:** Prompts are shared across all VLM profiles and preserved when switching models")
    
    # Get current prompts from prompts section (not profile-specific)
    prompts_config = vlm_config.get("prompts", {})
    oneshot_system = prompts_config.get("oneshot_system_prompt", "")
    oneshot_user = prompts_config.get("oneshot_user_prompt", "")
    
    # One-Shot Prompt Editing
    st.write("### 🎯 Single-Shot Comparison Prompts")
    st.caption("One image with both entry (left) and source (right) → AI returns scores directly")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**System Prompt:**")
        new_oneshot_system = st.text_area(
            "System instructions for one-shot comparison:",
            value=oneshot_system,
            height=200,
            help="Instructions for how the AI should compare the left (entered) and right (source) sides",
            key="oneshot_system"
        )
    
    with col2:
        st.write("**User Prompt:**")
        new_oneshot_user = st.text_area(
            "User message for one-shot comparison:",
            value=oneshot_user,
            height=200,
            help="The message sent with the comparison image. Should explain layout and expected response format.",
            key="oneshot_user"
        )
    
    # Prompt guidelines
    with st.expander("💡 Prompt Engineering Tips", expanded=False):
        st.markdown("""
**Key Elements for Effective Prompts:**

1. **Clear Layout Description:** "Left side is entered data, right side is source prescription"
2. **Specific Categories:** Define exactly what to compare (patient, prescriber, drug, direction)
3. **Scoring Scale:** "0-100 where 0=completely different, 100=identical"
4. **Response Format:** Show exact JSON format expected
5. **No Reasoning:** "Return ONLY JSON, no explanation"

**Example Response Format:**
```json
{
  "patient": 95,
  "prescriber": 100,
  "drug": 90,
  "direction": 85
}
```

**Scoring Guidelines to Include:**
- Same person/drug: 90-100
- Minor differences: 70-89
- Different person/drug: 0-20
- Missing data: 0
""")
    
    # Reset to defaults button
    col_reset, col_save = st.columns([1, 2])
    
    with col_reset:
        if st.button("🔄 Reset to Defaults", help="Reset prompts to default one-shot values"):
            # Set default prompts in the prompts section
            default_prompts = get_default_oneshot_prompts()
            if "prompts" not in vlm_config:
                vlm_config["prompts"] = {}
            vlm_config["prompts"].update(default_prompts)
            
            if save_vlm_config(vlm_config, config_file):
                st.success("✅ Prompts reset to defaults!")
                time.sleep(1)
                st.rerun()
    
    # Save prompts button
    with col_save:
        if st.button("💾 Save Prompts", type="primary"):
            # Update configuration in the prompts section (shared across all profiles)
            if "prompts" not in vlm_config:
                vlm_config["prompts"] = {}
            
            vlm_config["prompts"]["oneshot_system_prompt"] = new_oneshot_system
            vlm_config["prompts"]["oneshot_user_prompt"] = new_oneshot_user
            
            if save_vlm_config(vlm_config, config_file):
                st.success("✅ Prompts saved successfully!")
                st.info("🔄 Changes will take effect on the next VLM verification")
                time.sleep(1)
                st.rerun()

def get_default_oneshot_prompts() -> Dict[str, str]:
    """Get default prompts for one-shot VLM mode"""
    return {
        "oneshot_system_prompt": "You are a pharmacy prescription verification agent. Compare the prescription data and provide accurate matching scores.",
        
        "oneshot_user_prompt": """You are a pharmacy prescription verify agent, in the screenshot left side is what entered, right side is the original prescription. Please give me a matching score between 0-100 of patient, prescriber, drug, and direction. Only give score in following format, no reasoning, no explanation.
{
  "patient": score,
  "prescriber": score,
  "drug": score,
  "direction": score
}"""
    }

def show_model_settings(vlm_config: Dict[str, Any], config_file: str):
    """Show VLM model configuration settings with multiple profiles"""
    
    # Get current profile and profiles
    current_profile = vlm_config.get("current_profile", "online1")
    profiles = vlm_config.get("profiles", {})
    vlm_settings = vlm_config.get("vlm_settings", {})
    
    with st.expander("🤖 Model Configuration", expanded=True):
        
        # Profile selection
        st.write("**👥 VLM Profile Management:**")
        st.info("💡 **Multiple Profiles:** Save up to 3 different VLM configurations (local, online1, online2) for easy switching")
        
        profile_col1, profile_col2 = st.columns([3, 1])
        
        with profile_col1:
            profile_names = list(profiles.keys()) if profiles else ["local", "online1", "online2"]
            current_index = profile_names.index(current_profile) if current_profile in profile_names else 0
            
            selected_profile = st.selectbox(
                "Select VLM Profile:",
                options=profile_names,
                index=current_index,
                format_func=lambda x: f"{x.title()}: {profiles.get(x, {}).get('name', 'New Profile')}",
                help="Choose which VLM configuration to use and edit",
                key="profile_selector"
            )
        
        with profile_col2:
            if st.button("🔄 Switch Profile", help="Switch to selected profile"):
                vlm_config["current_profile"] = selected_profile
                if save_vlm_config(vlm_config, config_file):
                    st.success(f"✅ Switched to {selected_profile}!")
                    time.sleep(1)
                    st.rerun()
        
        # Current profile settings
        current_profile_config = profiles.get(selected_profile, {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**🔗 {selected_profile.title()} Profile Settings:**")
            
            profile_name = st.text_input(
                "Profile Display Name:",
                value=current_profile_config.get("name", f"{selected_profile.title()} Model"),
                help="Friendly name for this profile",
                key=f"{selected_profile}_name"
            )
            
            base_url = st.text_input(
                "Base URL:",
                value=current_profile_config.get("base_url", "http://localhost:1234/v1"),
                help="URL of your VLM API endpoint",
                key=f"{selected_profile}_url"
            )
            
            api_key = st.text_input(
                "API Key:",
                value=current_profile_config.get("api_key", "ollama"),
                type="password",
                help="API key or authentication token",
                key=f"{selected_profile}_key"
            )
        
        with col2:
            st.write("**⚙️ Model Parameters:**")
            
            model_name = st.text_input(
                "Model Name:",
                value=current_profile_config.get("model_name", "llava:7b"),
                help="Exact model name from your server",
                key=f"{selected_profile}_model"
            )
            
            max_tokens = st.number_input(
                "Max Tokens:",
                min_value=100,
                max_value=4000,
                value=current_profile_config.get("max_tokens", 1500),
                help="Maximum tokens in the response",
                key=f"{selected_profile}_tokens"
            )
            
            temperature = st.slider(
                "Temperature:",
                min_value=0.0,
                max_value=1.0,
                value=float(current_profile_config.get("temperature", 0.1)),
                step=0.1,
                help="Lower = more consistent, Higher = more creative",
                key=f"{selected_profile}_temp"
            )
    
        # Save profile button
        if st.button("💾 Save Profile Settings", type="primary", key="save_profile"):
            # Update the selected profile with new values
            if "profiles" not in vlm_config:
                vlm_config["profiles"] = {}
            
            vlm_config["profiles"][selected_profile] = {
                "name": profile_name,
                "base_url": base_url,
                "api_key": api_key,
                "model_name": model_name,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # Set as current profile if it wasn't already
            vlm_config["current_profile"] = selected_profile
            
            if save_vlm_config(vlm_config, config_file):
                st.success(f"✅ {selected_profile.title()} profile saved!")
                time.sleep(1)
                st.rerun()
        
        # Profile management buttons
        st.write("---")
        st.write("**🛠️ Profile Management:**")
        
        mgmt_col1, mgmt_col2, mgmt_col3 = st.columns(3)
        
        with mgmt_col1:
            if st.button("📋 Copy Profile", help="Copy current profile to another slot"):
                # Show modal or similar for copying profile
                st.info("💡 Feature coming soon: Copy profile functionality")
        
        with mgmt_col2:
            if st.button("🗑️ Reset Profile", help="Reset current profile to defaults"):
                default_configs = {
                    "local": {
                        "name": "Local Model",
                        "base_url": "http://localhost:1234/v1",
                        "api_key": "ollama",
                        "model_name": "llava:7b",
                        "max_tokens": 1500,
                        "temperature": 0.1
                    },
                    "online1": {
                        "name": "Google Gemini",
                        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                        "api_key": "your-api-key",
                        "model_name": "gemini-2.5-flash-lite-preview-09-2025",
                        "max_tokens": 1500,
                        "temperature": 0.1
                    },
                    "online2": {
                        "name": "OpenAI Compatible API",
                        "base_url": "http://localhost:1234/v1",
                        "api_key": "your-api-key",
                        "model_name": "gpt-4-vision-preview",
                        "max_tokens": 1500,
                        "temperature": 0.1
                    }
                }
                
                if selected_profile in default_configs:
                    vlm_config["profiles"][selected_profile] = default_configs[selected_profile]
                    if save_vlm_config(vlm_config, config_file):
                        st.success(f"✅ {selected_profile.title()} reset to defaults!")
                        time.sleep(1)
                        st.rerun()
        
        with mgmt_col3:
            if st.button("📤 Export Profile", help="Export current profile settings"):
                profile_export = {selected_profile: current_profile_config}
                st.download_button(
                    label="💾 Download Profile JSON",
                    data=json.dumps(profile_export, indent=2),
                    file_name=f"vlm_profile_{selected_profile}_{int(time.time())}.json",
                    mime="application/json"
                )
        
        # Image capture basics
        with st.expander("🖼️ Image Capture Settings", expanded=False):
            img_col1, img_col2 = st.columns(2)
            with img_col1:
                image_format = st.selectbox(
                    "Image Format:",
                    options=["PNG", "JPEG"],
                    index=0 if vlm_settings.get("image_format", "PNG") == "PNG" else 1,
                    help="Format for captured images"
                )
            with img_col2:
                image_quality = st.slider(
                    "Image Quality:",
                    min_value=50,
                    max_value=100,
                    value=vlm_settings.get("image_quality", 95),
                    help="Image quality (for JPEG format)"
                )



    # Quick access to coordinate GUI (outside collapsible for easy access)
    st.markdown("---")
    st.write("**🛠️ Coordinate Setup:**")
    if st.button("🎯 Open Settings GUI (settings_gui.py)"):
        try:
            # Path to settings_gui.py in ui folder
            script_dir = os.path.dirname(os.path.abspath(__file__))
            gui_script = os.path.join(script_dir, "settings_gui.py")
            if not os.path.exists(gui_script):
                st.error("❌ settings_gui.py not found in the UI folder")
            else:
                if sys.platform.startswith('win'):
                    subprocess.Popen([sys.executable, gui_script], creationflags=subprocess.CREATE_NEW_CONSOLE)
                else:
                    subprocess.Popen([sys.executable, gui_script])
                st.success("✅ Settings GUI launched! Check for a new window.")
        except Exception as e:
            st.error(f"❌ Failed to launch Settings GUI: {e}")

def show_region_setup(vlm_config: Dict[str, Any], config_file: str):
    """Show region setup for VLM screenshots"""
    st.subheader("📐 Screenshot Regions")
    
    st.info("🎯 **Setup Instructions:** Define the areas to capture for data entry and source prescription")
    
    vlm_regions = vlm_config.get("vlm_regions", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📝 Data Entry Region:**")
        st.caption("Area where prescription data is entered")
        
        data_entry_coords = vlm_regions.get("data_entry", [0, 0, 800, 600])
        
        de_x1 = st.number_input("X1 (Left):", value=data_entry_coords[0], key="de_x1")
        de_y1 = st.number_input("Y1 (Top):", value=data_entry_coords[1], key="de_y1")
        de_x2 = st.number_input("X2 (Right):", value=data_entry_coords[2], key="de_x2")
        de_y2 = st.number_input("Y2 (Bottom):", value=data_entry_coords[3], key="de_y2")
        
        new_data_entry = [de_x1, de_y1, de_x2, de_y2]
        
        if st.button("📸 Test Data Entry Capture", key="test_de"):
            test_screenshot(new_data_entry, "Data Entry")
    
    with col2:
        st.write("**📋 Source Prescription Region:**")
        st.caption("Area showing the original prescription")
        
        source_coords = vlm_regions.get("source", [800, 0, 1600, 600])
        
        src_x1 = st.number_input("X1 (Left):", value=source_coords[0], key="src_x1")
        src_y1 = st.number_input("Y1 (Top):", value=source_coords[1], key="src_y1")
        src_x2 = st.number_input("X2 (Right):", value=source_coords[2], key="src_x2")
        src_y2 = st.number_input("Y2 (Bottom):", value=source_coords[3], key="src_y2")
        
        new_source = [src_x1, src_y1, src_x2, src_y2]
        
        if st.button("📸 Test Source Capture", key="test_src"):
            test_screenshot(new_source, "Source")
    
    # Show current regions
    st.write("**📊 Current Regions:**")
    col3, col4 = st.columns(2)
    
    with col3:
        st.code(f"Data Entry: {new_data_entry}")
        de_width = new_data_entry[2] - new_data_entry[0]
        de_height = new_data_entry[3] - new_data_entry[1]
        st.caption(f"Size: {de_width}x{de_height}px")
    
    with col4:
        st.code(f"Source: {new_source}")
        src_width = new_source[2] - new_source[0]
        src_height = new_source[3] - new_source[1]
        st.caption(f"Size: {src_width}x{src_height}px")
    
    # Save regions
    if st.button("💾 Save Regions", type="primary"):
        vlm_config["vlm_regions"] = {
            "data_entry": new_data_entry,
            "source": new_source
        }
        
        if save_vlm_config(vlm_config, config_file):
            st.success("✅ Regions saved!")
            time.sleep(1)
            st.rerun()
    
    # Helper tools
    st.markdown("---")
    st.write("**🛠️ Helper Tools:**")
    
    col5, col6 = st.columns(2)
    
    with col5:
        if st.button("🖥️ Get Screen Size"):
            try:
                screen_width, screen_height = pyautogui.size()
                st.info(f"Screen size: {screen_width}x{screen_height}px")
            except Exception as e:
                st.error(f"Error getting screen size: {e}")
    
    with col6:
        if st.button("🎯 Launch Coordinate Helper", help="Opens visual coordinate selection tool"):
            launch_vlm_gui()
        st.caption("💡 Visual point-and-click coordinate selection")

def test_screenshot(coords: list, region_name: str):
    """Test screenshot capture for a region"""
    try:
        x1, y1, x2, y2 = coords
        width = x2 - x1
        height = y2 - y1
        
        if width <= 0 or height <= 0:
            st.error(f"❌ Invalid region dimensions for {region_name}")
            return
        
        # Capture screenshot
        screenshot = pyautogui.screenshot(region=(x1, y1, width, height))
        
        # Display the screenshot
        st.image(screenshot, caption=f"{region_name} Region Capture", use_column_width=True)
        st.success(f"✅ {region_name} capture successful!")
        
    except Exception as e:
        st.error(f"❌ Error capturing {region_name}: {e}")

def show_vlm_testing(vlm_config: Dict[str, Any]):
    """Show VLM testing interface"""
    st.subheader("🧪 VLM Testing")
    # Only two actions per requirement
    if st.button("🔍 Test VLM Connection", type="primary"):
        test_vlm_connection(vlm_config)
    if st.button("🚀 Run Complete VLM Verification", type="secondary"):
        run_complete_vlm_test(vlm_config)

def test_vlm_connection(vlm_config: Dict[str, Any]):
    """Test VLM connection"""
    try:
        from ai.vlm_verifier import VLM_Verifier
        
        verifier = VLM_Verifier(vlm_config)
        result = verifier.test_vlm_connection()
        
        if result.get("success"):
            st.success("✅ VLM Connection Successful!")
            st.write(f"**Model:** {result.get('model')}")
            st.write(f"**Base URL:** {result.get('base_url')}")
            st.write(f"**Response:** {result.get('response')}")
        else:
            st.error("❌ VLM Connection Failed!")
            st.write(f"**Error:** {result.get('error')}")
            st.write(f"**Model:** {result.get('model')}")
            st.write(f"**Base URL:** {result.get('base_url')}")
            
    except ImportError:
        st.error("❌ VLM Verifier module not available")
    except Exception as e:
        st.error(f"❌ Error testing VLM connection: {e}")

def capture_and_show_region(vlm_config: Dict[str, Any], region_name: str, display_name: str):
    """Capture and display a region screenshot"""
    try:
        from ai.vlm_verifier import VLM_Verifier
        
        verifier = VLM_Verifier(vlm_config)
        image = verifier.capture_region_screenshot(region_name)
        
        if image:
            st.image(image, caption=f"{display_name} Region", use_column_width=True)
            st.success(f"✅ {display_name} captured successfully!")
        else:
            st.error(f"❌ Failed to capture {display_name} region")
            
    except ImportError:
        st.error("❌ VLM Verifier module not available")
    except Exception as e:
        st.error(f"❌ Error capturing {display_name}: {e}")

def run_complete_vlm_test(vlm_config: Dict[str, Any]):
    """Run a complete VLM verification test - simply calls the main verify_with_vlm() function"""
    try:
        from ai.vlm_verifier import VLM_Verifier
        import logging
        import io
        
        st.info("🔄 Running complete VLM verification test...")
        
        # Create a custom logger to capture VLM debug output
        vlm_debug_log = io.StringIO()
        debug_handler = logging.StreamHandler(vlm_debug_log)
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        debug_handler.setFormatter(debug_formatter)
        
        # Get the VLM logger and add our handler
        vlm_logger = logging.getLogger('ai.vlm_verifier')
        original_level = vlm_logger.level
        vlm_logger.setLevel(logging.DEBUG)
        vlm_logger.addHandler(debug_handler)
        
        try:
            # Initialize verifier
            verifier = VLM_Verifier(vlm_config)
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("🤖 Running VLM verification (calling main verify_with_vlm function)...")
            progress_bar.progress(25)
            
            # Call the main verification function (this handles everything)
            scores = verifier.verify_with_vlm()
            
            progress_bar.progress(100)
            status_text.text("✅ VLM verification complete!")
            
            # Show results
            st.write("**🎯 Verification Results:**")
            
            if scores:
                # Display category scores - these are what the main app uses
                st.write("**📊 Category Scores:**")
                
                # Create columns dynamically based on number of categories
                num_categories = len(scores)
                cols = st.columns(num_categories) if num_categories > 0 else st.columns(1)
                
                for idx, (category, score) in enumerate(scores.items()):
                    # Prevent index out of range
                    if idx >= len(cols):
                        break
                        
                    with cols[idx]:
                        category_display = category.replace('_', ' ').title()
                        
                        # Color code the score
                        if score >= 90:
                            st.success(f"✅ **{category_display}**")
                            st.metric("", f"{score}%")
                        elif score >= 70:
                            st.warning(f"⚠️ **{category_display}**")
                            st.metric("", f"{score}%")
                        else:
                            st.error(f"❌ **{category_display}**")
                            st.metric("", f"{score}%")
                
                # Summary stats
                st.write("---")
                st.write("** Overall Summary:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    total_categories = len(scores)
                    st.metric("Total Categories", total_categories)
                
                with col2:
                    avg_score = sum(scores.values()) / total_categories if total_categories > 0 else 0
                    st.metric("Average Score", f"{avg_score:.1f}%")
                
                with col3:
                    high_scores = sum(1 for score in scores.values() if score >= 90)
                    st.metric("Passing Categories", f"{high_scores}/{total_categories}")
            else:
                st.warning("⚠️ No scores returned from VLM")
                st.info("This might indicate an issue with the model or configuration")
            
            # Show VLM debug output with all responses
            st.write("---")
            st.write("**🔍 VLM Debug Information:**")
            
            # Get the captured log output
            debug_output = vlm_debug_log.getvalue()
            
            # Parse and display VLM responses
            if debug_output:
                # Create expander for detailed debug log
                with st.expander("📋 Complete Debug Log", expanded=False):
                    st.code(debug_output, language="text")
                
                # Extract and highlight VLM responses
                st.write("**🤖 VLM Model Responses:**")
                
                # Extract LEFT image response (Step 1)
                if "LEFT extraction raw:" in debug_output:
                    left_start = debug_output.find("LEFT extraction raw:") + len("LEFT extraction raw:")
                    left_end = debug_output.find("DEBUG - LEFT extracted:", left_start)
                    if left_end == -1:
                        left_end = debug_output.find("\n", left_start + 200)
                    left_response = debug_output[left_start:left_end].strip()
                    
                    with st.expander("📤 Step 1: LEFT Image Extraction Response", expanded=True):
                        st.write("*What the VLM saw in the data entry (LEFT) image:*")
                        st.code(left_response, language="json")
                
                # Extract RIGHT image response (Step 2)
                if "RIGHT extraction raw:" in debug_output:
                    right_start = debug_output.find("RIGHT extraction raw:") + len("RIGHT extraction raw:")
                    right_end = debug_output.find("DEBUG - RIGHT extracted:", right_start)
                    if right_end == -1:
                        right_end = debug_output.find("\n", right_start + 200)
                    right_response = debug_output[right_start:right_end].strip()
                    
                    with st.expander("📥 Step 2: RIGHT Image Extraction Response", expanded=True):
                        st.write("*What the VLM saw in the source prescription (RIGHT) image:*")
                        st.code(right_response, language="json")
                
                # Extract scoring response (Step 3)
                if "Step3 raw response:" in debug_output:
                    score_start = debug_output.find("Step3 raw response:") + len("Step3 raw response:")
                    score_end = debug_output.find("DEBUG - Step3:", score_start)
                    if score_end == -1:
                        score_end = debug_output.find("\n", score_start + 200)
                    score_response = debug_output[score_start:score_end].strip()
                    
                    with st.expander("🎯 Step 3: Comparison & Scoring Response", expanded=True):
                        st.write("*How the VLM compared LEFT data with RIGHT data:*")
                        st.code(score_response, language="json")
                
                # Show any warnings or errors
                if "WARNING" in debug_output or "ERROR" in debug_output:
                    st.write("**⚠️ Warnings/Errors:**")
                    warnings = [line for line in debug_output.split('\n') if 'WARNING' in line or 'ERROR' in line]
                    for warning in warnings:
                        if 'ERROR' in warning:
                            st.error(warning)
                        else:
                            st.warning(warning)
            else:
                st.info("No debug output captured. Enable debug logging in VLM settings for detailed output.")
        
        finally:
            # Restore original logger settings
            vlm_logger.removeHandler(debug_handler)
            vlm_logger.setLevel(original_level)
            debug_handler.close()
            
    except ImportError:
        st.error("❌ VLM Verifier module not available")
    except Exception as e:
        st.error(f"❌ Error running VLM test: {e}")
        import traceback
        st.code(traceback.format_exc(), language="text")

def show_advanced_settings(vlm_config: Dict[str, Any], config_file: str):
    """Show advanced VLM settings"""
    st.subheader("📊 Advanced Settings")
    
    vlm_settings = vlm_config.get("vlm_settings", {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**🖼️ Image Processing:**")
        
        image_format = st.selectbox(
            "Image Format:",
            options=["PNG", "JPEG"],
            index=0 if vlm_settings.get("image_format", "PNG") == "PNG" else 1,
            help="Format for captured images"
        )
        
        image_quality = st.slider(
            "Image Quality:",
            min_value=50,
            max_value=100,
            value=vlm_settings.get("image_quality", 95),
            help="Image quality (for JPEG format)"
        )
        
        auto_enhance = st.checkbox(
            "Auto Enhance Images",
            value=vlm_settings.get("auto_enhance", True),
            help="Automatically enhance captured images"
        )
    
    with col2:
        st.write("**📏 Image Sizing:**")
        
        resize_max_width = st.number_input(
            "Max Width (px):",
            min_value=512,
            max_value=2048,
            value=vlm_settings.get("resize_max_width", 1024),
            help="Maximum image width (larger images will be resized)"
        )
        
        resize_max_height = st.number_input(
            "Max Height (px):",
            min_value=384,
            max_value=1536,
            value=vlm_settings.get("resize_max_height", 768),
            help="Maximum image height (larger images will be resized)"
        )

        st.write("**🧮 Image Token Budget (28px patches):**")

        max_image_tokens_total = st.number_input(
            "Max Image Tokens (total):",
            min_value=128,
            max_value=4096,
            value=int(vlm_settings.get("max_image_tokens_total", 1024)),
            step=64,
            help="Total image token budget per request. Roughly equals ceil(W/28)*ceil(H/28) summed across images. Increase for stronger GPUs."
        )

        default_per_image = max(64, int(vlm_settings.get("max_image_tokens_per_image", max_image_tokens_total // 2)))
        max_image_tokens_per_image = st.number_input(
            "Max Image Tokens (per image):",
            min_value=64,
            max_value=4096,
            value=min(default_per_image, max_image_tokens_total),
            step=64,
            help="Per-image token cap. Each image is resized to keep its tokens under this value."
        )

        if max_image_tokens_per_image > max_image_tokens_total:
            st.warning("Per-image token cap cannot exceed total tokens. It will be clamped on save.")
    
    # Optional maximum token safety cap
    st.markdown("---")
    st.write("**🛡️ Safety Settings:**")
    st.info("💡 **Optional Cap:** Set a maximum token limit to prevent excessive API costs or errors. Leave unchecked for no limit.")
    
    enable_cap = st.checkbox(
        "Enable Max Tokens Cap",
        value=vlm_settings.get("max_tokens_cap") is not None,
        help="Enable a safety limit on max_tokens to prevent runaway costs"
    )
    
    if enable_cap:
        max_tokens_cap = st.number_input(
            "Maximum Tokens Cap:",
            min_value=100,
            max_value=8000,
            value=vlm_settings.get("max_tokens_cap", 10000) if vlm_settings.get("max_tokens_cap") else 10000,
            step=100,
            help="Hard limit on max_tokens. System will warn and cap any profile that exceeds this value. Recommended: 3000-5000"
        )
        st.caption(f"🔍 **Example:** If profile has max_tokens=6000 and cap={max_tokens_cap}, system will use {max_tokens_cap} and log a warning.")
    else:
        max_tokens_cap = None
        st.caption("✅ **No limit:** Profiles can use any max_tokens value they specify.")
    
    # Save advanced settings
    if st.button("💾 Save Advanced Settings", type="primary"):
        vlm_config["vlm_settings"] = {
            "image_format": image_format,
            "image_quality": image_quality,
            "auto_enhance": auto_enhance,
            "resize_max_width": resize_max_width,
            "resize_max_height": resize_max_height,
            "max_image_tokens_total": int(max_image_tokens_total),
            "max_image_tokens_per_image": int(min(max_image_tokens_per_image, max_image_tokens_total)),
            "max_tokens_cap": max_tokens_cap,
            "_max_tokens_cap_help": "Optional safety limit for max_tokens. Set to null (no limit) or a number like 4000 to prevent excessive API costs. System will warn and cap if profile max_tokens exceeds this value."
        }
        
        if save_vlm_config(vlm_config, config_file):
            st.success("✅ Advanced settings saved!")
            time.sleep(1)
            st.rerun()
    
    # Configuration export/import
    st.markdown("---")
    st.write("**📤 Configuration Management:**")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if st.button("📥 Export VLM Config"):
            try:
                config_str = json.dumps(vlm_config, indent=2)
                st.download_button(
                    label="💾 Download VLM Config",
                    data=config_str,
                    file_name=f"vlm_config_backup_{int(time.time())}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    with col4:
        uploaded_file = st.file_uploader("📤 Import VLM Config", type="json")
        if uploaded_file is not None:
            try:
                imported_config = json.loads(uploaded_file.getvalue().decode("utf-8"))
                
                if save_vlm_config(imported_config, config_file):
                    st.success("✅ VLM configuration imported!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed to import configuration")
            except Exception as e:
                st.error(f"Import failed: {e}")
