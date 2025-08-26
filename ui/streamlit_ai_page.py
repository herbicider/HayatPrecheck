
import streamlit as st
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ai_config_page(app_config):
    st.header("🧠 AI Configuration")

    # Load existing AI settings from the main config
    ai_config = app_config.get("ai_config", {})

    # OpenAI compatible settings
    st.subheader("OpenAI Compatible API Settings")
    base_url = st.text_input("Base URL", value=ai_config.get("base_url", "https://api.openai.com/v1"))
    model_name = st.text_input("Model Name", value=ai_config.get("model_name", "gpt-4"))
    api_key = st.text_input("API Key", value=ai_config.get("api_key", ""), type="password")

    # Prompt customization
    st.subheader("Prompt Customization")
    
    # System Prompt
    default_system_prompt = """You are a pharmacist AI assistant verifying prescription data entry accuracy. 

Your task:
1. Compare entered vs source text for each field provided
2. Account for OCR errors, typos, and semantic equivalents
3. For drug names: different dosages = different drugs (metformin 500mg ≠ metformin 1000mg)
4. For names: preserve format importance (John Smith ≠ Smith John)
5. Score ALL fields provided in the user prompt

Response format (EXACTLY as shown):
patient_name: [0-100 score]
prescriber_name: [0-100 score] 
drug_name: [0-100 score]
direction_sig: [0-100 score]

Only include fields that are provided in the user prompt. Use whole numbers only. Process all available fields in a single response."""
    
    # Add example response format
    st.info("""
    **Expected AI Response Format:**
    ```
    patient_name: 85
    prescriber_name: 90
    drug_name: 75
    direction_sig: 30
    ```
    The AI should return scores (0-100) for each field on separate lines. Only include fields that are provided in the user prompt.
    """)
    
    system_prompt = st.text_area("System Prompt", value=ai_config.get("system_prompt", default_system_prompt), height=200)

    # --- User Prompt with Placeholders ---
    st.subheader("User Prompt")

    default_user_prompt = """Please verify all prescription fields:

Patient Name:
Entered: {patient_name_entered}
Source: {patient_name_source}

Prescriber Name:
Entered: {prescriber_name_entered}
Source: {prescriber_name_source}

Drug Name:
Entered: {drug_name_entered}
Source: {drug_name_source}

Directions/Sig:
Entered: {direction_sig_entered}
Source: {direction_sig_source}"""

    # Initialize session state for the prompt
    if 'ai_user_prompt' not in st.session_state:
        st.session_state.ai_user_prompt = ai_config.get("user_prompt", default_user_prompt)

    # --- Placeholders Section ---
    st.markdown("**Available Placeholders** (click to add to prompt)")
    
    enabled_optional = app_config.get("optional_fields_enabled", {})
    
    placeholders = {
        "Special": ["{field_to_verify}"],
        "Patient": ["{patient_name_entered}", "{patient_name_source}", "{patient_dob_entered}", "{patient_dob_source}", "{patient_address_entered}", "{patient_address_source}", "{patient_phone_entered}", "{patient_phone_source}"],
        "Prescriber": ["{prescriber_name_entered}", "{prescriber_name_source}", "{prescriber_address_entered}", "{prescriber_address_source}"],
        "Medication": ["{drug_name_entered}", "{drug_name_source}", "{direction_sig_entered}", "{direction_sig_source}"]
    }

    def is_placeholder_enabled(p):
        if "_dob" in p and not enabled_optional.get("patient_dob", False): return False
        if "patient_address" in p and not enabled_optional.get("patient_address", False): return False
        if "patient_phone" in p and not enabled_optional.get("patient_phone", False): return False
        if "prescriber_address" in p and not enabled_optional.get("prescriber_address", False): return False
        return True

    # Display buttons in columns
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for i, (category, ph_list) in enumerate(placeholders.items()):
        with cols[i]:
            st.markdown(f"**{category}**")
            for ph in ph_list:
                is_enabled = is_placeholder_enabled(ph)
                if st.button(ph, key=ph, use_container_width=True, disabled=not is_enabled):
                    st.session_state.ai_user_prompt += f" {ph}"

    # The text area now uses the session state
    st.session_state.ai_user_prompt = st.text_area(
        "User Prompt", 
        value=st.session_state.ai_user_prompt, 
        height=300,
        key="user_prompt_textarea"
    )

    if st.button("Save AI Configuration"):
        ai_config["base_url"] = base_url
        ai_config["model_name"] = model_name
        ai_config["api_key"] = api_key
        ai_config["system_prompt"] = system_prompt
        ai_config["user_prompt"] = st.session_state.ai_user_prompt
        
        # Remove old prompt key if it exists
        if "prompt" in ai_config:
            del ai_config["prompt"]

        try:
            with open("config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        config["ai_config"] = ai_config
        
        with open("config.json", "w") as f:
            json.dump(config, f, indent=4)
        st.success("AI configuration saved successfully!")

if __name__ == "__main__":
    # For direct testing, we create a dummy config
    dummy_config = {
        "optional_fields_enabled": {
            "patient_dob": True
        },
        "ai_config": {}
    }
    ai_config_page(dummy_config)
