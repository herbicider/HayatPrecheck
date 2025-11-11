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
- **Gemma3-12B-Vision** (best medical knowledge)
- **Qwen3-VL-8B** (fast, multilingual)


**Hardware Requirements:**
- Recommended: NVIDIA GPU with 16GB+ VRAM (inference ~2-4s/prescription)
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

**API Key Security (MANDATORY)**

```bash
# ✅ CORRECT - Store in .env file (git-ignored)
GEMINI_API_KEY=your_actual_key_here
OPENAI_API_KEY=sk-your_key_here
LOCAL_API_KEY=optional_local_auth_token
```

```json
// ✅ CORRECT - Reference in config files
{
  "api_key": "${GEMINI_API_KEY}"
}
```

**Never:**
- ❌ Hardcode API keys in `config.json` or `vlm_config.json`
- ❌ Commit `.env` file to version control
- ❌ Share API keys via email or messaging
- ❌ Store keys in screenshots or documentation

---

## 🚀 Complete Setup Guide for Non-Technical Users

### Step 1: Choose Your Hardware Setup

**For Non-Technical Staff:** You need a computer to run the AI verification. Here are your best options:

#### 💰 Best Bang for Buck: Mac Mini M4 (16GB) - $599-799
**Perfect for single pharmacy or small chain**

- **Specs**: Apple M4 chip, 16GB unified memory
- **Performance**: Runs 7B-12B vision models smoothly (~3-5 seconds per prescription)
- **Power**: Uses only 5-15 watts (saves electricity)
- **Noise**: Silent operation (no fans)
- **Setup**: Plug and play - macOS comes ready
- **HIPAA**: 100% local processing, patient data never leaves device
- **Lifespan**: 5-7 years of reliable service
- **Best Use**: 1-3 pharmacies, up to 600 prescriptions/day per pharmacy

**How to buy:**
1. Visit [apple.com/mac-mini](https://www.apple.com/mac-mini/)
2. Choose M4 model with **16GB RAM minimum** (24GB even better)
3. Order directly or buy from Best Buy, Amazon

#### 🔥 Sweet Spot for Multi-Location: AMD Ryzen AI 9 HX 395 - $1,800-2,200
**Best for 5-10 pharmacy locations**

- **Specs**: AMD Ryzen AI 9 HX 395 processor with NPU
- **Performance**: Handles 14B+ models, ~2-3 seconds per prescription
- **Scalability**: Can serve multiple locations via network
- **Power**: ~50-100 watts under load
- **Setup**: Windows 11 Pro (familiar for most users)
- **HIPAA**: Local server behind your firewall
- **Best Use**: 5-10 pharmacies, centralized processing hub

**Where to buy:**
- **Pre-built systems**: HP Elite, Lenovo ThinkStation with AMD AI processors
- **Custom builds**: Work with local IT vendor to specify AMD AI 9 HX 395
- **Online**: Newegg, B&H Photo, Amazon Business

#### ❌ What NOT to Buy (Overkill for Pharmacies)

**NVIDIA DGX Station (~$4,000+)**
- Enterprise AI workstation
- Designed for training models, not running them
- Massive overkill for pharmacy verification
- High power consumption and cooling requirements

**Multi-RTX Card Builds ($5,000-15,000)**
- Gaming/enthusiast setups
- Unnecessary complexity and cost
- Higher failure rates, more maintenance
- Power hungry (300-800 watts)

**Why Mac Mini or AMD AI is Better:**
- **Cost**: 5-20x cheaper than GPU workstations
- **Efficiency**: Modern AI chips designed for inference
- **Reliability**: Fewer moving parts, longer lifespan
- **Maintenance**: Zero maintenance vs constant driver updates
- **Noise**: Silent vs loud fans

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
   - **Windows**: Press `Windows Key + R`, type `cmd`, press Enter
   - **Mac**: Press `Command + Space`, type `terminal`, press Enter
   - Type: `python --version`
   - You should see: `Python 3.12.x` or similar

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

1. **Open Command Prompt/Terminal:**
   - **Windows**: Press `Windows Key + R`, type `cmd`, Enter
   - **Mac**: Press `Command + Space`, type `terminal`, Enter

2. **Navigate to the folder:**
   ```bash
   # Windows example (adjust path to where you extracted):
   cd C:\Users\YourName\Desktop\HayatPrecheck
   
   # Mac example:
   cd ~/Desktop/HayatPrecheck
   ```

3. **Install the system:**
   ```bash
   # This installs everything needed automatically
   pip install -r requirements.txt
   ```
   
   **Wait 2-5 minutes** for installation to complete. You'll see lots of text scrolling - this is normal!

#### D. Run the System (Daily Use)

**Every time you want to use the verification system:**

1. **Open Command Prompt/Terminal**
2. **Navigate to folder:**
   ```bash
   cd C:\Users\YourName\Desktop\HayatPrecheck
   # or on Mac:
   cd ~/Desktop/HayatPrecheck
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

**That's it!** The system will guide you through setup on first run.

#### E. Quick Start Shortcut (Windows Only)

**Create a desktop shortcut so you don't need Command Prompt:**

1. Right-click on Desktop → New → Shortcut
2. Enter location:
   ```
   C:\Users\YourName\AppData\Local\Programs\Python\Python312\python.exe "C:\Users\YourName\Desktop\HayatPrecheck\launcher.py"
   ```
3. Name it: "Pharmacy Verification"
4. Click Finish
5. **Double-click this shortcut to start** the system anytime!

---

### Step 3: AI Hardware Setup (Local HIPAA-Compliant)

**For Mac Mini M4 or AMD AI Systems:**

#### Option A: Ollama (Recommended - Easiest)

1. **Download Ollama:**
   - Visit [https://ollama.ai](https://ollama.ai)
   - Click "Download for Mac" or "Download for Windows"
   - Install like any normal application

2. **Install a Vision Model:**
   - Open Terminal/Command Prompt
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

#### Option 1: Google Gemini (Recommended - Best Medical Knowledge)

**Getting API Key with HIPAA BAA:**

1. **Sign up for Google Cloud:**
   - Visit [https://cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai)
   - Click "Get Started"
   - Enter business information

2. **Request BAA (Business Associate Agreement):**
   - Contact Google Cloud Sales: [https://cloud.google.com/contact](https://cloud.google.com/contact)
   - Tell them: "I need HIPAA compliance for healthcare application"
   - Legal team will send you BAA to sign
   - **DO NOT use API until BAA is signed!**

3. **Get API Key:**
   - Once BAA is signed, go to: [https://console.cloud.google.com](https://console.cloud.google.com)
   - Navigate to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy your API key (starts with `AIza...`)

4. **Enable Gemini API:**
   - In Google Cloud Console, search for "Gemini API"
   - Click "Enable"
   - Wait 1-2 minutes

5. **Configure in System:**
   - In your HayatPrecheck folder, create file named `.env`
   - Add this line:
     ```
     GEMINI_API_KEY=AIzaYourActualKeyHere
     ```
   - Save file
   - Open web dashboard → VLM Configuration
   - Select "Gemini" profile
   - Test connection

**Pricing:**
- Free tier: 15 requests/minute
- Paid: ~$0.0025 per prescription verification
- For 500 prescriptions/day: ~$35/month

#### Option 2: OpenAI GPT-4 Vision (Alternative)

**Getting API Key with HIPAA BAA:**

1. **Sign up for OpenAI:**
   - Visit [https://platform.openai.com/signup](https://platform.openai.com/signup)
   - Create account with business email

2. **Request BAA:**
   - Email: [https://help.openai.com/](https://help.openai.com/)
   - Subject: "HIPAA BAA Request for Pharmacy Application"
   - They will send BAA documents
   - **Do not use until BAA is signed**

3. **Get API Key:**
   - Visit [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - Click "Create new secret key"
   - Name it: "Pharmacy Verification"
   - Copy key (starts with `sk-...`)
   - **Save immediately** - you can't view it again!

4. **Add Billing:**
   - Go to [https://platform.openai.com/account/billing](https://platform.openai.com/account/billing)
   - Add credit card
   - Set monthly limit: $100-500 depending on volume

5. **Configure in System:**
   - Create `.env` file in HayatPrecheck folder
   - Add:
     ```
     OPENAI_API_KEY=sk-YourActualKeyHere
     ```
   - Save file
   - Open web dashboard → VLM Configuration
   - Select "OpenAI" profile
   - Model: `gpt-4-vision-preview`
   - Test connection

**Pricing:**
- ~$0.01-0.02 per prescription verification
- For 500 prescriptions/day: ~$150-300/month

#### Option 3: Anthropic Claude (Alternative)

**Getting API Key with HIPAA BAA:**

1. **Sign up:**
   - Visit [https://www.anthropic.com/claude](https://www.anthropic.com/claude)
   - Click "Get API Access"

2. **Request BAA:**
   - Contact [privacy@anthropic.com](mailto:privacy@anthropic.com)
   - Subject: "HIPAA BAA Request"
   - Include: business name, use case, expected volume

3. **Get API Key:**
   - Once approved: [https://console.anthropic.com/](https://console.anthropic.com/)
   - Go to "API Keys"
   - Create new key
   - Copy key (starts with `sk-ant-...`)

4. **Configure:**
   - Add to `.env` file:
     ```
     ANTHROPIC_API_KEY=sk-ant-YourKeyHere
     ```

**Pricing:**
- ~$0.015 per verification
- For 500 prescriptions/day: ~$225/month

---

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
   - ✅ Microsoft Azure (OpenAI Service) - [BAA Info](https://www.microsoft.com/en-us/trust-center/compliance/hipaa)

4. **Providers WITHOUT BAA = HIPAA Violation:**
   - ❌ Generic OpenAI API (without healthcare agreement)
   - ❌ Consumer AI services (ChatGPT, Bard, etc.)
   - ❌ Any provider that won't sign BAA

**Cost Comparison for 500 Rx/day:**

| Option | Initial Cost | Monthly Cost | BAA Required? |
|--------|--------------|--------------|---------------|
| Mac Mini M4 16GB | $499 one-time | $0 | No (local) |
| AMD AI System | $2,000 one-time | $0 | No (local) |
| Google Gemini API | $0 | $35-50 | Yes |
| OpenAI GPT-4 Vision | $0 | $150-300 | Yes |
| Anthropic Claude | $0 | $225 | Yes |

**Recommendation:**
- **1-3 pharmacies**: Mac Mini M4 (best value)
- **5-10 pharmacies**: AMD AI system as central server
- **10+ pharmacies**: Cloud API with BAA (operational simplicity)

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

### AI/VLM Issues

**Problem: Model not responding**
- Verify local AI server is running (Ollama/LM Studio)
- Test endpoint in VLM Configuration page
- Check model name matches exactly
- Review logs for detailed error messages

**Problem: Low accuracy scores**
- Customize prompts in Streamlit UI page
- Use chain-of-thought prompting technique! 


**"API Key not found"**
```bash
# Create .env file in project root
# Add: GEMINI_API_KEY=your_key_here
# Never put API keys in config.json files!
```

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
    ├── settings_gui.py     # Coordinate setup
    └── vlm_gui.py          # VLM configuration
```

---

## Development Roadmap

**✅ Completed (Current v2.0)**
- Single-shot VLM verification (3x faster than multi-step)
- OpenAI-compatible API support with multiple profiles
- Local AI deployment for HIPAA compliance
- Production testing with Gemma3-12B, Qwen2.5-VL-7B

**🚧 In Progress**
- Enhanced prompt templates library
- Multi-model ensemble verification
- Automatic region detection using computer vision

**🔮 Future Phases**
- Custom pharmacy-trained models
- Enterprise multi-location deployment
- Advanced analytics and predictive error detection
- Mobile tablet integration

---

## License & Contributing

**License**: MIT (see LICENSE file)

**Contributing**: Issues and pull requests welcome! Please review `AGENTS.md` for development guidelines.

**Development Focus:**
- HIPAA compliance and privacy-first design
- Performance optimization for resource-constrained environments
- User experience and ease of setup
- Compatibility with various pharmacy software systems

---

**Last Updated**: November 2025  
