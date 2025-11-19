# Pharmacy Pre-Check Verification Agent

**Production-ready prescription verification system combining traditional OCR with cutting-edge AI vision technology for HIPAA-compliant pharmacy automation.**

## Overview

This system automates pharmacy data entry verification using two complementary approaches:

1. **Traditional OCR + Fuzzy Matching** - Fast, reliable baseline (no AI required)
2. **Vision Language Models (VLM)** - AI-powered verification for complex prescriptions and handwriting


## Key Features

**Dual Verification Modes**
- Traditional OCR + fuzzy matching (fast, no AI dependencies)
- VLM single-shot verification (AI vision for handwriting and complex layouts)

**HIPAA-Compliant Architecture**
- 100% local processing option (patient data stays on-premises)
- Offline operation capability

**User-Friendly Interface**
- Web-based Streamlit dashboard
- Point-and-click coordinate setup

---

## 🔒 HIPAA Compliance & Privacy

### Critical Privacy Considerations

**⚠️ PATIENT DATA SECURITY WARNING**

When using AI/VLM features, understand where patient data is processed:

#### ✅ HIPAA-COMPLIANT OPTIONS (Recommended)

**1. Local AI Deployment (Best Practice)**
- **Ollama** - Easiest local LLM deployment ([ollama.ai](https://ollama.ai))
- **LM Studio** - User-friendly local model management ([lmstudio.ai](https://lmstudio.ai))


**Benefits:**
- ✅ Patient data NEVER leaves your premises
- ✅ No internet required for AI verification
- ✅ Full HIPAA compliance
- ✅ Complete control over data processing
- ✅ No recurring API costs

**Recommended Local Models:**
- **Gemma3-12B** (best medical knowledge)


**Hardware Requirements:**
- Recommended: NVIDIA GPU with 16GB+ VRAM 
- Apple Silicon: M1/M2/M3 with 16GB+ unified memory works well

**2. On-Premises API Server**
- Deploy OpenAI-compatible API server behind your firewall


#### ⚠️ NON-COMPLIANT OPTIONS (Use with Caution)

**Cloud AI APIs (OpenAI, Anthropic, Google Gemini)**
- ❌ Patient data transmitted over internet
- ❌ Data processed on third-party servers
- ❌ Potential HIPAA violation without BAA
- ⚠️ Requires Business Associate Agreement (BAA)
- ⚠️ May violate state privacy laws

**When Cloud APIs Might Be Acceptable:**
1. Vendor provides signed BAA (Business Associate Agreement)
2. Data is de-identified before transmission (risky, hard to guarantee)

### HIPAA Compliance Checklist

Before deploying in production:

- [ ] **Choose deployment method**: Local models (recommended) or cloud with BAA
- [ ] **Configure .env file**: Never hardcode API keys in config files
- [ ] **Enable audit logging**: Track all verification activities
- [ ] **Test offline operation**: Verify system works without internet
- [ ] **Review data flows**: Ensure patient data stays local
- [ ] **Document procedures**: Create compliance documentation
- [ ] **Train staff**: Ensure team understands privacy settings
- [ ] **Regular audits**: Monitor logs for unexpected data transmission

### Secure Configuration

**API Key Security**

```bash
# ✅ CORRECT - Store in .env file (git-ignored)
GEMINI_API_KEY=your_actual_key_here
OPENAI_API_KEY=sk-your_key_here
LOCAL_API_KEY=optional_local_auth_token
```

## 🚀 Complete Setup Guide

### Step 1: Choose Your Hardware Setup

**For Non-Technical Staff:** You need a computer to run the local AI model. Here are your best options:

#### 💰 Best Bang for Buck: Windows PC with NVIDIA GPU - $800-1,500
**Perfect for single pharmacy or small chain**

- **Specs**: NVIDIA RTX 4060 or higher (8GB+ VRAM), 16GB RAM
- **Performance**: Runs 7B-12B vision models smoothly (~2-4 seconds per prescription)
- **Setup**: Install Ollama on Windows, runs in background
- **Why**: Pharmacy software already runs on Windows - use same computer


#### 🔥 Sweet Spot for Multi-Location: AMD Ryzen AI or Intel Core Ultra - $1,500-2,500
**Best for 5-10 pharmacy locations**

- **Specs**: AMD Ryzen AI 9 HX 395 or Intel Core Ultra 9, 32-64GB RAM, RTX 4070+
- **HIPAA**: Local Windows server behind your firewall
- **Best Use**: 5-10 pharmacies, centralized processing hub
- **Advantage**: Dedicated AI processing, handles multiple pharmacy locations

#### ❌ What NOT to Buy (Overkill for Pharmacies)

**NVIDIA DGX Spark Station (~$4,000+)**
- Designed for training models, not running them


**Multi-RTX Card Builds ($5,000-15,000)**
- Unnecessary complexity and cost


**Why Modern Windows AI Hardware is Better:**
- **Cost**: 5-20x cheaper than enterprise GPU servers
- **Integration**: Runs on same PC as pharmacy software - no extra hardware
- **Efficiency**: RTX 4000-series GPUs designed for AI inference
- **Reliability**: Standard Windows PC - easy to replace/upgrade
- **Maintenance**: Minimal - Ollama auto-updates in background

#### 🌐 For Large Chains (10+ Locations): Cloud AI with BAA

Instead of expensive hardware, use cloud APIs with Business Associate Agreement:

- **Best for**: 10+ pharmacy locations, enterprise deployments
- **Cost**: Pay-per-use (~$0.001-0.005 per prescription verification)
- **Benefits**: No hardware to maintain, scales infinitely, always up-to-date
- **Requirement**: Must have signed BAA (Business Associate Agreement) with provider

---

### Step 2: Running the Code (For Non-Programmers)

**Complete First-Time Setup - Follow These Steps Exactly:**

#### A. Install Python (One-Time Setup)

1. **Download Python:**
   - Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
   - Click the big yellow button "Download Python 3.12.x"
   
2. **Install Python:**
   - **CRITICAL**: Check the box "Add Python to PATH" at the bottom
   - Click "Install Now"
   - Wait for installation to complete
   - Click "Close"

3. **Verify Installation:**
   - Press `Windows Key + R`, type `cmd`, press Enter
   - Type: `python --version`
   - You should see: `Python 3.12.x` or similar
   - If not found, restart your computer and try again

#### B. Download the Verification System

**Option 1: Download ZIP (Easiest for Non-Programmers)**
1. Go to [https://github.com/herbicider/HayatPrecheck](https://github.com/herbicider/HayatPrecheck)
2. Click green "Code" button → "Download ZIP"
3. Extract the ZIP file to your Desktop or Documents folder
4. Remember this location!

**Option 2: Use Git (For Technical Users)**
```bash
git clone https://github.com/herbicider/HayatPrecheck.git
cd HayatPrecheck
```

#### C. Install Required Software (One-Time Setup)

1. **Open Command Prompt:**
   - Press `Windows Key + R`, type `cmd`, Enter

2. **Navigate to the folder:**
   ```bash
   # Adjust path to where you extracted the ZIP:
   cd C:\Users\YourName\Desktop\HayatPrecheck
   ```

3. **Install the system:**
   ```bash
   # This installs everything needed automatically
   pip install -r requirements.txt
   ```
   
   **Wait 2-5 minutes** for installation to complete. You'll see lots of text scrolling - this is normal!

#### D. Run the System (Daily Use)

**Every time you want to use the verification system:**

#### 🚀 EASIEST WAY (Recommended for Windows):

1. **Navigate to the HayatPrecheck folder**
2. **Double-click `start.bat`**
3. **That's it!** The system launches automatically

The `start.bat` file:
- Automatically activates Python environment
- Launches the menu system
- No command prompt needed!
- No typing required!

#### Alternative: Using Command Prompt

1. **Open Command Prompt**
2. **Navigate to folder:**
   ```bash
   cd C:\Users\YourName\Desktop\HayatPrecheck
   ```

3. **Start the launcher:**
   ```bash
   python launcher.py
   ```

4. **Choose an option from the menu:**
   - Option 1: Setup coordinates (first time only)
   - Option 2: Start verification
   - Option 3: Open web dashboard
   - Option 4: Configure AI/VLM settings

#### E. Create Desktop Shortcut (Optional)

**For even faster access:**

1. Right-click on `start.bat` in the HayatPrecheck folder
2. Choose "Create shortcut"
3. Drag the shortcut to your Desktop
4. Rename it: "Pharmacy Verification"
5. **Double-click anytime to start!**

---

### Step 3: AI Hardware Setup (Local HIPAA-Compliant)

**For Windows PCs with NVIDIA GPU:**

#### Option A: Ollama (Recommended - Easiest)

1. **Download Ollama:**
   - Visit [https://ollama.ai](https://ollama.ai)
   - Click "Download for Windows"
   - Install like any normal application
   - Ollama runs in background automatically

2. **Install a Vision Model:**
   - Open Command Prompt (Windows Key + R, type `cmd`)
   - Type: `ollama pull llava:13b`
   - Wait 5-10 minutes for download (model is ~8GB)

3. **Start Ollama:**
   - It starts automatically after installation
   - You'll see an Ollama icon in your system tray
   - The model runs on `http://localhost:11434`

4. **Configure in Pharmacy System:**
   - Run: `streamlit run ui/streamlit_app.py`
   - Go to "VLM Configuration" page
   - Set API URL: `http://localhost:11434/v1`
   - Set Model Name: `qwen3-vl:8b`
   - Click "Test Connection"

**Recommended Models for Pharmacy:**
```bash
# Best accuracy (requires 16GB+ RAM):
ollama pull gemma3:12b

# Multilingual support:
ollama pull qwen3-vl:8b
```

#### Option B: LM Studio (User-Friendly GUI)

1. **Download LM Studio:**
   - Visit [https://lmstudio.ai](https://lmstudio.ai)
   - Click "Download LM Studio"
   - Install the application

2. **Download a Vision Model:**
   - Open LM Studio
   - Go to "Discover" tab
   - Search: "Gemma 3 12b"
   - Click download (wait 10-15 minutes)

3. **Start the Server:**
   - Go to "Local Server" tab
   - Click "Start Server"
   - Note the URL: usually `http://localhost:1234`

4. **Configure in Pharmacy System:**
   - Set API URL: `http://localhost:1234/v1`
   - Set Model Name: (shown in LM Studio)
   - Test connection

---

### Step 4: Cloud AI Setup with BAA (For Scaling)

**When to Use Cloud AI:**
- You have 10+ pharmacy locations
- You want to avoid hardware management
- You have a signed Business Associate Agreement (BAA)


### Important Notes on Cloud APIs and HIPAA

**⚠️ CRITICAL: BAA (Business Associate Agreement) Requirements**

1. **What is a BAA?**
   - Legal contract between you (pharmacy) and AI provider
   - Provider agrees to protect patient health information (PHI)
   - Required by HIPAA law for any third-party handling PHI

2. **Before Using Cloud APIs:**
   - [ ] Contact provider sales/legal team
   - [ ] Request and review BAA documents
   - [ ] Have your pharmacy lawyer review BAA
   - [ ] Sign BAA with authorized pharmacy representative
   - [ ] Keep signed copy for compliance audits
   - [ ] **Only then** use API for patient data

3. **Providers that Offer BAA:**
   - ✅ Google Cloud (Gemini) - [BAA Info](https://cloud.google.com/security/compliance/hipaa)
   - ✅ OpenAI (GPT-4) - Contact via support
   - ✅ Anthropic (Claude) - Contact sales


4. **Providers WITHOUT BAA = HIPAA Violation:**


**Cost Comparison for 500 Rx/day:**

| Option | Initial Cost | Monthly Cost | BAA Required? |
|--------|--------------|--------------|---------------|
| Mac Mini M4 16GB | $499 one-time | $0 | No (local) |
| AMD AI System | $2,000 one-time | $0 | No (local) |
| Google Gemini API | $0 | $35-50 | Yes |
| OpenAI GPT | $0 | $150-300 | Yes |
| Anthropic Claude | $0 | $225 | Yes |

---

## Usage

### Web Dashboard (Primary Interface)

```bash
After run the launch.bat

# Access at: http://localhost:8501

 If no new page pop up
```

**Dashboard Features:**
- UI for easy setup

### Verification Modes

**Mode 1: Traditional OCR**

**Mode 2: VLM Single-Shot** (Recommended)


### Automation Options

**Manual Verification Mode:**
- Visual feedback only (green/red field highlights)
- Review verification results
- No automatic actions

**Autopilot Mode:**
- Automatically sends configured key when all fields match
- Configurable delay and key selection (F1-F12, Enter, etc.)
- Safety confirmation period

### Configuration Files

- **`config/config.json`** - Main settings, thresholds, automation
- **`config/vlm_config.json`** - VLM model settings and prompts
- **`config/abbreviations.json`** - Pharmacy term expansions
- **`.env`** - API keys and secrets (NEVER commit to git)

---

## Project Architecture

```
📁 HayatPrecheck/
├── launcher.py              # Main entry point - START HERE
├── requirements.txt         # Python dependencies
├── .env                     # API keys (create this, git-ignored)
│
├── 📁 config/              # Configuration files
│   ├── config.json         # Main settings
│   ├── vlm_config.json     # VLM/AI settings
│   └── abbreviations.json  # Pharmacy terms
│
├── 📁 core/                # Verification engine
│   ├── verification_controller.py  # Main logic
│   ├── comparison_engine.py        # Field matching
│   └── ocr_provider.py             # OCR management
│
├── 📁 ai/                  # AI/ML modules
│   └── vlm_verifier.py     # Vision model integration
│
└── 📁 ui/                  # User interfaces
    ├── streamlit_app.py    # Web dashboard
    └── settings_gui.py     # Unified coordinate setup (OCR & VLM)
```

---

## Development Roadmap

**✅ Completed (Current v2.0)**
- Single-shot VLM verification (3x faster than multi-step)
- OpenAI-compatible API support with multiple profiles
- Local AI deployment for HIPAA compliance
- Production testing with Gemma3-12B, Qwen2.5-VL-7B

**🔮 Future Phases**
- Custom pharmacy fine-tuned models
- Enterprise multi-location deployment


---

## License & Contributing

**License**: MIT 

**Contributing**: Issues and pull requests welcome!

**Last Updated**: November 2025  
