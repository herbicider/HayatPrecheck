
import streamlit as st
import json
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def ai_config_page(app_config):
    st.header("🧠 LLM AI Configuration")
    st.info("⚡ **Configure LLM AI for OCR + LLM verification method**")

    # Load LLM configuration from llm_config.json
    try:
        with open("config/llm_config.json", "r") as f:
            config = json.load(f)
            llm_config = config.get("llm_config", {})
    except FileNotFoundError:
        config = {}
        llm_config = {}

    # LLM API Settings
    st.subheader("🔗 LLM API Settings")
    st.info("💡 **Tip**: Works with OpenAI API, LMStudio, Ollama, or any OpenAI-compatible endpoint")
    
    base_url = st.text_input("Base URL", value=llm_config.get("base_url", "http://localhost:1234/v1"),
                            help="LMStudio: http://localhost:1234/v1 | Ollama: http://localhost:11434/v1")
    model_name = st.text_input("Model Name", value=llm_config.get("model_name", ""),
                              help="Leave empty for LMStudio. For Ollama use model name like 'llama3.2'")
    api_key = st.text_input("API Key", value=llm_config.get("api_key", ""), type="password",
                           help="Not needed for local LLMs (LMStudio/Ollama)")

    # Advanced LLM Settings  
    with st.expander("⚙️ Advanced LLM Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_tokens = st.number_input("Max Tokens", min_value=100, max_value=2000, 
                                       value=llm_config.get("max_tokens", 500))
        with col2:
            temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, 
                                        value=llm_config.get("temperature", 0.1), step=0.1)

    # Prompt Customization for OCR + LLM Method
    st.subheader("📝 Prompt Customization for OCR + LLM")
    st.info("🎯 **These prompts are used by the LLM AI verification method (OCR + LLM comparison)**")
    
    # System Prompt
    default_system_prompt = llm_config.get("system_prompt", """You are a pharmacy prescription verification assistant. Compare entered prescription data with source prescription data. CRITICAL: Respond with ONLY valid JSON. No explanations, no markdown, no extra text. Use whole numbers only (0-100).

Verification Rules:
1. Verify entered vs source text for accuracy and equivalency
2. Account for OCR artifacts and common transcription errors
3. Give equivalency score for drugs (brand name = generic)
4. Matching one of the source prescribers (if there are 2, separated by '/') is a match
5. Use semantic analysis for directions - focus on meaning, not exact wording
6. Consider pharmacy abbreviations and medical terminology
7. If entered data exists but source is empty/missing -> Score = 0
8. If source data exists but entered is empty/missing -> Score = 0

SCORING GUIDELINES:
- 95-100: Perfect/near perfect match
- 85-94: Very good match with minor differences
- 75-84: Good match, acceptable differences  
- 60-74: Questionable match, needs review
- 0-59: Poor match, significant discrepancies, or missing data""")
    
        # Add example response format for OCR + LLM
    st.info("""
    **Expected LLM Response Format (JSON):**
    ```json
    {
      "patient_name": 85,
      "prescriber_name": 90,
      "drug_name": 95,
      "direction_sig": 80
    }
    ```
    """)
    
    system_prompt = st.text_area("System Prompt", value=default_system_prompt, height=200)

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
        st.session_state.ai_user_prompt = default_user_prompt

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

    # ===== FIELD EXTRACTION PATTERNS CONFIGURATION =====
    st.subheader("🔍 Field Extraction Patterns")
    st.info("🎯 **Customize OCR field detection patterns** - Configure how the LLM verifier identifies different fields in your pharmacy software")
    
    # Load current field patterns from config
    field_patterns = config.get("field_extraction_patterns", {})
    
    # Create tabs for different fields
    pattern_tabs = st.tabs(["👤 Patient Name", "👨‍⚕️ Prescriber Name", "💊 Drug Name", "📋 Directions/Sig"])
    
    # Patient Name Patterns
    with pattern_tabs[0]:
        st.markdown("**Patient Name Detection Patterns**")
        patient_config = field_patterns.get("patient_name", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Primary Patterns** (high priority)")
            patient_primary = st.text_area(
                "Primary patterns (one per line)", 
                value="\n".join(patient_config.get("primary_patterns", ["Patient:", "patient:", "Patient Name:", "Name:"])),
                height=120,
                key="patient_primary",
                help="Patterns like 'Patient:' that commonly precede patient names"
            )
        
        with col2:
            st.markdown("**Fallback Patterns** (backup)")
            patient_fallback = st.text_area(
                "Fallback patterns (one per line)", 
                value="\n".join(patient_config.get("fallback_patterns", ["pt:", "PT:", "Pt Name:"])),
                height=120,
                key="patient_fallback",
                help="Alternative patterns to try if primary patterns fail"
            )
    
    # Prescriber Name Patterns
    with pattern_tabs[1]:
        st.markdown("**Prescriber Name Detection Patterns**")
        prescriber_config = field_patterns.get("prescriber_name", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Primary Patterns**")
            prescriber_primary = st.text_area(
                "Primary patterns (one per line)", 
                value="\n".join(prescriber_config.get("primary_patterns", ["Written By:", "written by:", "Issued By:", "Prescriber:", "Doctor:"])),
                height=120,
                key="prescriber_primary",
                help="Patterns like 'Written By:' that commonly precede prescriber names"
            )
        
        with col2:
            st.markdown("**Fallback Patterns**")
            prescriber_fallback = st.text_area(
                "Fallback patterns (one per line)", 
                value="\n".join(prescriber_config.get("fallback_patterns", ["Dr.", "Physician:", "MD:", "Supervisor:", "Signed electronically by:"])),
                height=120,
                key="prescriber_fallback",
                help="Alternative patterns including credentials and signatures"
            )
    
    # Drug Name Patterns  
    with pattern_tabs[2]:
        st.markdown("**Drug Name Detection Patterns**")
        drug_config = field_patterns.get("drug_name", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Primary Patterns**")
            drug_primary = st.text_area(
                "Primary patterns (one per line)", 
                value="\n".join(drug_config.get("primary_patterns", ["Item:", "item:", "Medication:", "Drug:"])),
                height=120,
                key="drug_primary",
                help="Patterns like 'Item:' that commonly precede drug names"
            )
        
        with col2:
            st.markdown("**Fallback Patterns**")
            drug_fallback = st.text_area(
                "Fallback patterns (one per line)", 
                value="\n".join(drug_config.get("fallback_patterns", ["Med:", "Rx:", "Medicine:", "Prescribed:"])),
                height=120,
                key="drug_fallback",
                help="Alternative patterns for medication identification"
            )
    
    # Directions/Sig Patterns
    with pattern_tabs[3]:
        st.markdown("**Directions/Sig Detection Patterns**")
        sig_config = field_patterns.get("direction_sig", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Primary Patterns**")
            sig_primary = st.text_area(
                "Primary patterns (one per line)", 
                value="\n".join(sig_config.get("primary_patterns", ["Directions:", "directions:", "Directions (Sig Codes or Text or Literal Text ):", "Sig:", "Instructions:"])),
                height=120,
                key="sig_primary",
                help="Patterns like 'Directions:' that commonly precede dosing instructions"
            )
        
        with col2:
            st.markdown("**Fallback Patterns**")
            sig_fallback = st.text_area(
                "Fallback patterns (one per line)", 
                value="\n".join(sig_config.get("fallback_patterns", ["Take:", "Use:", "Apply:", "How to use:"])),
                height=120,
                key="sig_fallback",
                help="Action words that start dosing instructions"
            )
    
    st.markdown("---")
    
    if st.button("Save AI Configuration"):
        # Save LLM config
        llm_config["base_url"] = base_url
        llm_config["model_name"] = model_name
        llm_config["api_key"] = api_key
        llm_config["system_prompt"] = system_prompt
        llm_config["user_prompt"] = st.session_state.ai_user_prompt
        
        # Remove old prompt key if it exists
        if "prompt" in llm_config:
            del llm_config["prompt"]

        # Save field extraction patterns
        updated_field_patterns = {
            "patient_name": {
                "primary_patterns": [p.strip() for p in patient_primary.split('\n') if p.strip()],
                "fallback_patterns": [p.strip() for p in patient_fallback.split('\n') if p.strip()],
                "description": "Patterns to identify patient name in OCR text"
            },
            "prescriber_name": {
                "primary_patterns": [p.strip() for p in prescriber_primary.split('\n') if p.strip()],
                "fallback_patterns": [p.strip() for p in prescriber_fallback.split('\n') if p.strip()],
                "description": "Patterns to identify prescriber name in OCR text"
            },
            "drug_name": {
                "primary_patterns": [p.strip() for p in drug_primary.split('\n') if p.strip()],
                "fallback_patterns": [p.strip() for p in drug_fallback.split('\n') if p.strip()],
                "description": "Patterns to identify drug name in OCR text"
            },
            "direction_sig": {
                "primary_patterns": [p.strip() for p in sig_primary.split('\n') if p.strip()],
                "fallback_patterns": [p.strip() for p in sig_fallback.split('\n') if p.strip()],
                "description": "Patterns to identify directions/sig in OCR text"
            }
        }

        # Load existing config or create new one
        try:
            with open("config/llm_config.json", "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            config = {}
        
        # Update config with new values
        config["llm_config"] = llm_config
        config["field_extraction_patterns"] = updated_field_patterns
        
        # Save to file
        with open("config/llm_config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        st.success("✅ AI configuration and field extraction patterns saved successfully!")
        st.info("🔄 **Note:** Changes will take effect when the LLM verifier is next initialized.")

if __name__ == "__main__":
    # For direct testing, we create a dummy config
    dummy_config = {
        "optional_fields_enabled": {
            "patient_dob": True
        },
        "ai_config": {}
    }
    ai_config_page(dummy_config)
