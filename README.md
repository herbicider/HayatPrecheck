# Pharmacy Pre-Check Verification Agent

## 🆕 What's New?

**10/21/2025: Single-Shot VLM Verification - Revolutionary Simplification!** 🚀

**⚡ One Image, One Prompt, Instant Results**: After discovering that 12B+ models (like Gemma 3-12B) can handle direct side-by-side comparison, the entire VLM system has been redesigned for maximum efficiency:

**What Changed**:
- **Before**: 5-step process (capture left → extract → capture right → extract → compare) = 3 API calls
- **After**: 1-step process (capture comparison area → get scores) = 1 API call
- **Performance**: 3x faster, 3x cheaper, simpler code, better accuracy!

**How It Works Now**:
1. System captures ONE screenshot showing both pharmacy entry (left) and original prescription (right)
2. Sends single prompt: *"Compare left vs right, give me matching scores"*
3. AI returns JSON scores directly: `{patient: 95, prescriber: 100, drug: 90, direction: 85}`

**Benefits**:
- 🚀 **Speed**: 2-4 seconds vs 6-10 seconds (3x faster)
- 💰 **Cost**: 1/3 the token usage
- 🎯 **Accuracy**: AI sees both images simultaneously for better visual comparison
- 🛠️ **Simplicity**: One region to configure, one prompt to tune
- ✅ **Reliability**: Fewer failure points, more consistent results

**Setup**: Just use `vlm_gui.py` to select ONE region containing both sides, and you're done! See `ONESHOT_MIGRATION.md` for complete migration guide.

**Recommended Models**: Gemma 3-12B, Qwen2.5-VL-14B, or any 12B+ vision-language model. 7B models may still work but 12B+ provides superior accuracy with the simplified approach.

---

**10/01/2025: Major VLM and LLM Enhancements** 

**🎯 Enhanced Multi-Step VLM Verification** *(Now superseded by single-shot approach above)*: 
Upgraded to RTX5080 16GB to run Qwen2.5-VL-7B Q8 via LM Studio. Rerouted to the Vision Language Model system with a sophisticated 4-step analysis process:
- **Step 1**: AI extracts visible text from data entry
- **Step 2**: AI extract visible text from original prescription image
- **Step 3**: Performs field-by-field comparison with semantic understanding
- **Step 4**: Provides confidence scores (0-100) with detailed reasoning

*Note: The single-shot approach (10/21/2025) replaces this multi-step process for 12B+ models, offering significantly better performance.*

**🔧 JSON Format Enforcement**: Both VLM and LLM systems now feature standard JSON parsing:
- Handles AI response variations and formatting inconsistencies  
- Ensures reliable field scoring and verification results
- Unified JSON architecture across all AI verification methods

**Medically trained LLM**: Currently using a medically trained Gemma3 4B model for LLM semantic analysis. In the future I plan to train a Hayat specific model.

**08/25/2025: Vision Language Model (VLM) Integration** 

**🎯 VLM-Powered Direct Image Verification**: The system now supports Vision Language Models for direct prescription image comparison without OCR text extraction. This revolutionary approach analyzes screenshots directly using AI vision capabilities, providing superior accuracy for handwritten prescriptions and complex layouts while maintaining visual context and formatting. Supports multiple models including Qwen2-VL, LLaVA-Next, and Phi-3.5-Vision for flexible deployment options. All AIs use the OpenAI API standard for seamless integration. I'm using Qwen2.5-VL-7B Q4_K_M on A770 8GB, all prompts are designed for this model.

**08/23/2025: Major Improvement in OCR Performance!**

**Asynchronous Processing**: The system now supports asynchronous processing for improved performance and responsiveness. Also the image pre-process is standardized regardless of the OCR engine selection. This allows for faster verification and reduced waiting times, especially when dealing with multiple prescriptions in a row. *Now it's as fast as I have to add delay timer for the next scan*. However, please be patient with the first time turning ON the monitor or the very first scan since it's loading a lot of libraries.

**08/21/2025: Adaptive Trigger for Other Software**

**Customized Trigger**: Instead of hardcoding the trigger to "Pre-Check Rx", the system now allows you to set a custom trigger area and keyword to start the verifying process. Also a separated "Rx#" area for detecting new prescriptions. I have made all settings available in the config.json file and accessible on the streamlit settings page for easy configuration. This allows you to adapt the system to work with any pharmacy software that has a similar pre-check verification process.

**08/19/2025: AI Incorporation** (Enhanced 10/09/2025)

**Enhanced AI-Powered Verification**: The system integrates with OpenAI-compatible APIs for intelligent text comparison and semantic matching, now featuring **unified JSON formatting** for consistency with VLM systems. Configure your preferred AI endpoint, API key, and model through the dedicated AI settings page. 

**New JSON Architecture Features**:
- **Structured Response Format**: All LLM responses now use standardized JSON format
- **Enhanced Prompt Engineering**: Strict JSON enforcement prevents parsing errors
- **Unified Processing**: Same robust parsing system as VLM for reliability
- **Backward Compatibility**: Works seamlessly with existing OCR workflows
- **Field-Level Control**: Choose AI vs traditional fuzzy matching per field type

**Tested Model Performance**:
- **Mistral-7B-Q4-K_M** via llamacpp (recommended): Perfect balance of speed and accuracy
- **Phi-3 Mini**: Excellent for HIPAA-compliant local deployment
- **Qwen2.5**: Superior multilingual and pharmaceutical terminology understanding
- **Gemma3**: Cost-effective options for basic verification tasks

**Hardware Requirements**: Successfully runs on Mac Mini M1 8GB, providing excellent cost-effectiveness for small pharmacies. The system can handle CRM, databases, file servers, and AI models simultaneously.

**🔒 Enhanced HIPAA Compliance**: For maximum privacy protection, use local AI models (Phi-3 mini, Gemma 3, or Qwen3 via llamacpp) instead of cloud-based APIs. The new JSON architecture ensures no patient data parsing errors while maintaining complete local processing.

**08/15/2025: Major OCR Engine Update**

**PaddleOCR Removed**: After extensive testing, PaddleOCR has been removed from the application due to poor performance with small text areas. It's reserved for future full page VLM analysis. 

**Current OCR Strategy**: The application now uses a dual-engine approach with **automatic selection**:
- **EasyOCR**: Primary choice for GPU-enabled systems (better accuracy)
- **Tesseract**: Fallback for CPU-only systems (faster performance, requires separate installation)

I've noticed on CPU only computer, Tesseract is faster than EasyOCR while EasyOCR provides better accuracy. Pick at your own discretion.

*Have ideas or suggestions? Please submit an issue or reach out!*

---

**AI-Powered Prescription Verification System**

Elevate pharmacy data entry verification with cutting-edge AI technology! This system combines traditional OCR with revolutionary Vision Language Models (VLM) to automate prescription verification, allowing pharmacists to focus on clinical decision-making rather than repetitive data validation.

## 🌟 Key Features

### 🚀 Advanced AI Integration
- **Single-Shot VLM Verification**: Revolutionary one-image, one-prompt comparison (12B+ models)
- **Vision Language Models (VLM)**: Direct image analysis without OCR text extraction
- **3x Performance Boost**: Single API call vs multi-step process for instant results
- **Local AI Support**: HIPAA-compliant local models (Ollama, LM Studio, LlamaFile)
- **Intelligent OCR Selection**: Automatic GPU detection for optimal performance

### 🎯 Smart Verification
- **Real-time Monitoring**: Visual feedback with colored field overlays (green/red)
- **Automated Processing**: Optional autopilot mode with configurable key sending
- **Semantic Understanding**: AI-powered pharmaceutical terminology recognition
- **Customizable Thresholds**: Adjustable accuracy settings per field type

### 🖥️ User-Friendly Interface
- **Web Dashboard**: Streamlit-based monitoring and configuration
- **Visual Setup Tools**: Point-and-click coordinate selection
- **Mobile Access**: Monitor verification from any device
- **Comprehensive Analytics**: Performance tracking and error analysis

### 🔒 Privacy & Compliance
- **100% Local Processing**: Patient data never leaves your premises
- **HIPAA Compliant**: Designed for healthcare environments
- **No Internet Required**: Core functionality works offline
- **Open Source**: Complete transparency and auditability  


## 🔧 How It Works

The system monitors your pharmacy software and performs intelligent verification:

### Verification Process
1. **Trigger Detection**: Monitors for customizable trigger keywords/areas
2. **Data Capture**: Screenshots comparison region (both data entry and source visible)
3. **AI Analysis**: Choice of three verification methods:
   - **VLM Mode (Single-Shot)**: One screenshot → One prompt → Instant scores (12B+ models)
   - **AI + OCR Mode**: OCR extraction followed by semantic AI comparison
   - **Traditional Mode**: OCR with fuzzy string matching
4. **Visual Feedback**: Colored overlays indicate field matches (green) or mismatches (red)
5. **Automation**: Optional autopilot mode sends configurable key presses
6. **Continuous Monitoring**: Automatically detects new prescriptions via Rx number changes

### Intelligent Text Processing

#### AI-Powered Analysis (Recommended)
- **Semantic Understanding**: AI comprehends pharmaceutical terminology and context
- **Multi-Step Verification**: Separate text extraction, comparison, and confidence scoring
- **Robust JSON Parsing**: 5-strategy system eliminates format errors
- **Field-Level Control**: Choose verification method per field type

#### Traditional Methods (Fallback)
- **Smart Text Cleaning**: Handles name formats, titles, abbreviations
- **Pharmaceutical Intelligence**: Extensive abbreviation expansion system
- **Advanced Drug Matching**: Dosage validation and salt form handling
- **Flexible Sig Processing**: Accounts for different direction formats

## 🤖 AI-Powered Verification Technology

### Vision Language Model (VLM) Integration

**Revolutionary Single-Shot Direct Image Analysis**
- **No OCR Required**: AI "sees" images directly without text extraction
- **One Image, One Prompt**: Captures both sides simultaneously for instant comparison
- **3x Faster**: Single API call (2-4 seconds) vs multi-step process (6-10 seconds)
- **Superior Accuracy**: AI maintains visual context for better comparison
- **Best for Handwriting**: Excellent with complex layouts and handwritten prescriptions
- **12B+ Models**: Optimized for Gemma 3-12B, Qwen2.5-VL-14B, and similar

### AI Verification Methods

#### Option 1: VLM Mode - Single-Shot (Recommended for 12B+ Models)
**Direct Side-by-Side Comparison**
- One Screenshot (both sides) → One Prompt → JSON Scores
- 3x faster and cheaper than multi-step approaches
- Best for handwritten prescriptions and complex layouts
- Maintains full visual context throughout analysis
- Supports Gemma 3-12B, Qwen2.5-VL-14B+, LLaVA-Next, Phi-3.5-Vision models
- Configure with `vlm_gui.py` - select single comparison region

#### Option 2: LLM + OCR Mode
**Enhanced Traditional Approach**
- OCR Text Extraction → LLM Semantic Comparison → Confidence Scoring
- Backward compatible with existing workflows
- Field-level AI vs traditional matching control
- Works with any OpenAI-compatible API

#### Option 3: Traditional Mode (Fallback)
**String-Based Matching**
- OCR extraction with fuzzy string matching
- Pharmaceutical abbreviation expansion
- Smart text cleaning and normalization
- Fastest processing option

### 🔧 AI System Features

#### Robust JSON Processing
**5-Strategy Parsing System**:
1. Direct JSON parsing
2. Markdown code block extraction
3. JSON boundary detection
4. Text cleaning and retry
5. Regex pattern matching

#### Enhanced Error Handling
- **Missing Data Detection**: Intelligent scoring of incomplete fields
- **Response Validation**: Ensures proper field structure and data types
- **Graceful Degradation**: Automatic fallback when AI responses fail
- **Comprehensive Testing**: 100% success rate validation

#### HIPAA Compliance
- **Local AI Models**: Ollama, LlamaFile, or LM Studio deployment
- **No Internet Required**: Complete offline operation capability
- **Patient Data Protection**: All processing stays on premises
- **Audit Trails**: Complete verification logging

## 🚀 Quick Start Guide

### Step 1: Installation

#### Download & Extract
1. Click the **`< > Code`** button → **`Download ZIP`**
2. Extract to your desired location (e.g., Desktop)

#### Python Setup
1. Install Python from [python.org](https://www.python.org/downloads/)
2. ⚠️ **Critical**: Check "Add Python to PATH" during installation

#### Install Dependencies
```bash
cd HayatPrecheck
pip install -r requirements.txt
```

**For Advanced Users** (Virtual Environment):
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Step 2: Automatic Setup (Recommended)

**🎯 One-Command Launch:**
```bash
python launcher.py
```

The launcher automatically:
- ✅ Detects GPU availability and selects optimal OCR provider
- ✅ Installs EasyOCR if needed
- ✅ Configures system settings
- ✅ Tests installation
- ✅ Launches setup wizard

### Step 3: Choose Your Verification Method

#### Option A: Vision Language Model - Single-Shot (Best Accuracy & Speed)
1. Install local AI server (LM Studio/Ollama recommended)
2. Load 12B+ VLM model (Gemma 3-12B or Qwen2.5-VL-14B recommended)
3. Run visual region selector: `python ui/vlm_gui.py`
4. Select ONE comparison region containing both data entry (left) and prescription (right)
5. Configure model endpoint via Streamlit VLM Configuration page
6. See `ONESHOT_MIGRATION.md` for detailed setup guide

#### Option B: Traditional OCR + AI
1. Set up OpenAI-compatible API endpoint
2. Configure via Streamlit AI Settings page  
3. Choose per-field AI vs traditional matching

#### Option C: Traditional OCR Only
1. Use coordinate setup GUI
2. Configure thresholds via Streamlit
3. Pure string-based matching

### Step 4: Configuration

#### Coordinate Setup (GUI Method)
```bash
python ui/settings_gui.py
```
- Visual drag-and-drop region selection
- Real-time OCR testing
- Automatic validation

#### Web Interface Configuration
```bash
streamlit run ui/streamlit_app.py
```
Open: `http://localhost:8501`
- Complete settings management
- Real-time monitoring
- Performance analytics

### OCR Engine Details

#### Intelligent Auto-Selection
The system automatically chooses the best OCR provider:

| Hardware | Primary Choice | Backup |
|----------|----------------|--------|
| NVIDIA GPU | EasyOCR (CUDA) | Tesseract |
| CPU Only | Tesseract | EasyOCR (CPU) |
| Apple Silicon | EasyOCR (MPS) | Tesseract |

#### Manual OCR Installation
**Tesseract** (if needed):
- **Windows**: [Download installer](https://digi.bib.uni-mannheim.de/tesseract/)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

## �️ User Interface Options

### 🌐 Web Dashboard (Primary Interface)

**Launch Options:**
```bash
# Auto-launcher (recommended)
python launcher.py  # Choose option 3

# Direct launch
streamlit run ui/streamlit_app.py
```

**Access:** `http://localhost:8501`

**Features:**
- 📊 **Real-Time Monitoring**: Live verification results and performance charts
- ⚙️ **Complete Configuration**: All settings accessible through web interface
- 📱 **Mobile Responsive**: Monitor from any device on your network
- 🎯 **Visual Feedback**: Field-by-field analysis and color-coded results
- 📈 **Analytics Dashboard**: Performance metrics and accuracy tracking
- 🔧 **Testing Tools**: OCR and VLM testing directly in browser

### 🖼️ GUI Configuration Tools

#### Coordinate Setup GUI
```bash
python ui/settings_gui.py
```

**Features:**
- **Visual Selection**: Screenshot with drag-and-drop region selection  
- **Live Testing**: Real-time OCR verification for each region
- **Precise Control**: Zoom, keyboard shortcuts, coordinate validation
- **Complete Settings**: Thresholds, automation, and general configuration

#### VLM Visual Configuration (Single-Shot Mode)
```bash
python ui/vlm_gui.py
```

**Features:**
- **Single Region Selection**: Select ONE area containing both sides
- **Point-and-Click Setup**: Visual comparison region selection
- **Live Preview**: Real-time screenshot capture with overlay
- **Green Color Coding**: Single comparison region highlighted
- **Coordinate Validation**: Automatic region testing and validation
- **Simple Setup**: Just select the area showing both pharmacy entry (left) and prescription (right)

### 📋 Configuration Management

#### Matching Thresholds
Control verification strictness per field:
- **Patient Name**: Default 85% similarity
- **Prescriber Name**: Default 85% similarity  
- **Drug Name**: Default 85% similarity
- **Directions/Sig**: Default 85% similarity

**Tuning Guidelines:**
- **Higher values** → More strict (fewer false positives)
- **Lower values** → More lenient (catch more variations)

#### Automation Settings
- **Auto-Key Sending**: Configurable key press when all fields match
- **Supported Keys**: F1-F12, Enter, Tab, Space, Escape
- **Timing Control**: Adjustable delay before key sending
- **Safety Features**: Visual confirmation periods

### 🔧 Basic Operations

**Start Monitoring:**
```bash
python core/verification_controller.py
```

**Stop Program:**
- Press `Ctrl + C` in terminal
- Use Stop button in Streamlit interface
- Close application window

## 🆘 Troubleshooting

### Common Issues & Solutions

#### OCR Problems
**Issue**: Text not being detected properly
- **Solution**: Use GUI coordinate tools to ensure regions are precise
- **Tip**: Avoid boxes, lines, or artifacts - select text-only areas
- **Test**: Use "Test OCR" feature to verify each region
- **Fallback**: System automatically retries with configurable intervals

#### Web Interface Issues
**Issue**: Streamlit won't open
```bash
# Alternative launch methods
python -m streamlit run ui/streamlit_app.py
# Try different addresses
http://127.0.0.1:8501
http://localhost:8501
```
- Check firewall settings (port 8501)
- Try incognito mode or different browser
- Ignore HTTPS warnings (Streamlit uses HTTP by default)

#### Tesseract Path Errors
**Issue**: "tesseract is not installed or not in PATH"
- **Quick Fix**: Set `"ocr_provider": "auto"` in config.json
- **Windows PATH Fix**:
  1. Find Tesseract install location (`C:\Program Files\Tesseract-OCR`)
  2. Add to Windows PATH environment variable
  3. Restart Command Prompt

#### AI/VLM Connection Issues
**Issue**: AI models not responding
- **Check**: Model server is running (Ollama/LM Studio)
- **Verify**: API endpoint and model name in configuration
- **Test**: Use testing tabs in Streamlit VLM/AI Configuration pages
- **Troubleshoot**: Check logs for detailed error messages

### Performance Optimization

#### Speed Improvements
1. **Use GPU acceleration** when available (EasyOCR with CUDA)
2. **Minimize screenshot regions** to essential text areas only
3. **Enable VLM quantization** (Q4_K_M recommended)
4. **Adjust polling intervals** based on pharmacy software speed

#### Accuracy Tuning
1. **Start with trigger region** - get this working first
2. **Test each field individually** before full verification
3. **Adjust thresholds** based on your data entry patterns
4. **Use pharmacy-specific prompts** for AI models
5. **Monitor logs** to identify common error patterns

### 🎯 Success Tips

#### Setup Best Practices
- **Clear screenshots**: Ensure text is clearly visible
- **Minimal regions**: Select smallest areas containing target text
- **Test immediately**: Verify OCR after each region selection
- **Validate configuration**: Use built-in validation tools before monitoring

#### Optimization Guidelines
- **Threshold tuning**: Adjust based on pharmacy workflow differences
- **Prompt engineering**: Customize AI prompts for better accuracy
- **Hardware utilization**: Use GPU when available for better performance
- **Regular monitoring**: Check logs for patterns and improvement opportunities

## 🔒 Privacy & Security

### HIPAA Compliance Features
- ✅ **100% Local Processing**: Patient data never leaves premises
- ✅ **Offline Operation**: No internet required for core functionality
- ✅ **Local AI Models**: Ollama, LlamaFile support for complete privacy
- ✅ **Audit Trails**: Comprehensive logging for compliance requirements
- ✅ **Open Source**: Full transparency and code auditability

### Security Best Practices
- **Use local AI models** for sensitive data processing
- **Configure firewalls** to restrict external AI model access
- **Regular backups** of configurations and logs
- **Monitor access logs** for security audit requirements
- **Update dependencies** regularly for security patches

## � Project Structure

### File Organization
```
📁 HayatPrecheck/
├── 🚀 launcher.py              # Main entry point - start here!
├── 📄 requirements.txt         # Dependencies
├── 📖 README.md               # This documentation
├── 📁 config/                 # Configuration files
│   ├── config.json           # Main application settings
│   ├── vlm_config.json       # VLM model configuration
│   ├── llm_config.json       # AI/LLM settings
│   └── abbreviations.json    # Pharmacy abbreviations
├── 📁 core/                   # Core verification engine
│   ├── verification_controller.py  # Main verification logic
│   ├── comparison_engine.py        # Field comparison algorithms
│   └── ocr_provider.py             # OCR engine management
├── 📁 ai/                     # AI and ML modules
│   ├── vlm_verifier.py       # Vision Language Model engine
│   └── cpu_verifier.py       # CPU-based AI text verification
├── 📁 ui/                     # User interfaces
│   ├── streamlit_app.py      # Main web dashboard
│   ├── settings_gui.py       # GUI coordinate setup
│   └── vlm_gui.py           # VLM visual configuration
└── 📁 tools/                  # Testing and utilities
```

### Quick Access Points

**🚀 Getting Started:**
- **Main Launcher**: `python launcher.py`
- **Web Dashboard**: `streamlit run ui/streamlit_app.py`
- **Coordinate Setup**: `python ui/settings_gui.py`

**🔧 Configuration:**
- **General Settings**: `config/config.json`
- **VLM Setup**: `config/vlm_config.json` + `ui/vlm_gui.py`
- **AI Configuration**: `config/llm_config.json`

**🧪 Testing:**
- **OCR Testing**: Built into GUI tools
- **VLM Testing**: Built into web interface and GUIs
- **Full System Test**: Use launcher's test mode

## 🚀 Future Development Roadmap

### ✅ Completed Milestones

**Single-Shot VLM Revolution (October 2025)**
- One image, one prompt, instant results
- 3x performance improvement over multi-step approach
- Optimized for 12B+ vision-language models
- Simplified configuration with single comparison region

**Vision Language Model Integration (August 2025)**
- Direct image analysis without OCR extraction
- Multi-step AI verification with confidence scoring (superseded by single-shot)
- Visual coordinate selection tools
- Local deployment for HIPAA compliance

### � Next Phase: Advanced AI Architecture

#### Phase 1: Enhanced Intelligence (Q1 2026)
- **Multi-Model Ensemble**: Consensus-based verification using multiple AI models
- **Custom Pharmacy Training**: Fine-tuned models for prescription-specific accuracy
- **Adaptive Region Detection**: Computer vision for automatic coordinate setup
- **Predictive Error Analysis**: AI-powered error pattern recognition

#### Phase 2: Enterprise Scalability (Q2-Q3 2026)
- **Centralized Processing Hub**: Single powerful server serving multiple pharmacies
- **Network Architecture**: Secure, scalable multi-location deployment
- **Advanced Analytics**: Cross-pharmacy insights and quality metrics
- **Regulatory Compliance**: Enhanced audit trails and reporting tools

#### Phase 3: Next-Generation Features (Q4 2026+)
- **Full Prescription Analysis**: Complete document understanding beyond field verification
- **Contextual Intelligence**: Drug interaction and dosage validation integration
- **Mobile Integration**: Tablet and smartphone verification capabilities
- **Cloud-Hybrid Options**: Secure cloud processing with local fallback

### 🌟 Technology Evolution

**Previous State**: OCR → Multi-Step AI Extraction → Comparison (5 steps, 3 API calls)  
**Current State**: Single-Shot VLM → Direct Comparison → Instant Scores (1 step, 1 API call)  
**Future Vision**: Image → Multi-AI Consensus → Clinical Intelligence → Automated Quality Assurance

### 📚 Documentation

**Quick Start Guides:**
- `README.md` - This comprehensive guide
- `ONESHOT_MIGRATION.md` - **NEW**: Complete single-shot VLM migration guide
- `AGENTS.md` - Repository guidelines and development docs

**Key Features:**
- Single-shot VLM verification setup and testing
- Performance comparison (old vs new approach)
- Configuration examples and prompt engineering tips
- Troubleshooting and optimization guides

### Contributing & Development

This project represents the cutting edge of pharmacy automation technology. The modular architecture makes it easy to extend and customize for specific pharmacy workflows.

**Development Priorities:**
1. **AI Model Optimization**: Improving accuracy and reducing resource requirements
2. **Single-Shot VLM Enhancement**: Further optimization for 12B+ models
3. **User Experience**: Streamlined setup and configuration processes  
4. **Scalability**: Enterprise deployment and multi-location support
5. **Integration**: APIs and connectors for popular pharmacy management systems

*The future of pharmacy verification is AI-powered, and this project is leading the transformation.*
