# What's new?

**08/25/2025: Vision Language Model (VLM) Integration** (Testing and improving)

**🎯 VLM-Powered Direct Image Verification**: The system now supports Vision Language Models for direct prescription image comparison without OCR text extraction. This revolutionary approach analyzes screenshots directly using AI vision capabilities, providing superior accuracy for handwritten prescriptions and complex layouts while maintaining visual context and formatting. Configure your VLM server (I'm running Qwen2-VL on a 8Gb Vram PC) and enable visual point-and-click region selection through the new VLM Configuration GUI.

**08/23/2025: Major Improvement in OCR performance!**

**Asynchronous Processing**: The system now supports asynchronous processing for improved performance and responsiveness. Also the image pre-process is standardized regardless of the OCR engine selection. This allows for faster verification and reduced waiting times, especially when dealing with multiple prescriptions in a row. *Now it's as fast as I have to add delay timer for the next scan*. However, please be patient with the first time turning ON the monitor or the very first scan since it's loading a lot of libraries.

**08/21/2025: Adaptive trigger for other software**

**Customized Trigger**: Instead of hardcoding the trigger to "Pre-Check Rx", the system now allows you to set a custom trigger area and keyword to star the verifying process. Also a separated "Rx#" area for detecting new prescriptions. I have made all settings to the config.json file and availabe on the streamlit settings page for easy configuration. This allows you to adapt the system to work with any pharmacy software that has a similar pre-check verification process.

**08/19/2025: AI incorporation**

**AI-Powered Verification**: The system now is able to integrate with OpenAI-compatible APIs for intelligent text comparison and semantic matching. Configure your preferred AI endpoint, API key, and model through the dedicated AI settings page. Customize system and user prompts to optimize verification accuracy for your specific workflow. It allows you to choos what to use AI and what to use traditional fuzzy matching on a per-field basis for maximum flexibility and HIPAA compliance (NEVER send patient info to online AI). 

I tried *Gemma 3 1b* model but the accuracy was not satisfying, now I'm running *Llama 3.2 3b Q4 K M* model via *llamacpp* on a Mac mini M1 8Gb. The 3b model fits the server capability and provided perfect balance between speed and accuracy. The default prompt and settings are set around this model. You can use a more powerful PC as the AI server but I'm trying to keep the cost low and efficient for small business. A used M1 Mac Mini 8Gb is only about $250, it's running my PHP8 for CRM, MySQL, sFTP, all python reports and now the local AI model fine. Well bang for the buck!

**🔒 HIPAA Compliance Notice**: For maximum privacy protection, use local AI models (like Phi-3 mini, Gemma 3, or Qwen3 via llamacpp) instead of cloud-based APIs when processing patient data. During development and testing, I tested with Phi-3 mini to ensure sensitive information never leaves the premises.

**08/15/2025: Major OCR Engine Update**

**PaddleOCR Removed**: After extensive testing, PaddleOCR has been removed from the application due to poor performance with small text areas. It's reserved for future full page VLLM analysis. 

**Current OCR Strategy**: The application now uses a dual-engine approach with **automatic selection**:
- **EasyOCR**: Primary choice for GPU-enabled systems (better accuracy)
- **Tesseract**: Fallback for CPU-only systems (faster performance, requires separate installation)

I've noticed on CPU only computer, Tesseract is faster than EasyOCR while EasyOCR provides better accuracy. Pick at your own discretion.

*Have ideas or suggestions? Please submit an issue or reach out!* 


# Pharmacy Pre-Check Verification Agent

The **simplest** way to set up and assit your pharmacy data entry verification system! 

Quality pharmacist time should be utilized in a more professional task like therapeutic review, drug-drug-interaction identfication, solution to personalized patient needs, instead of mechanically reading and comparing for the data entry accurcay. 

Human gets tired after hours of reading and, unavoidly, the error rate goes up. While a machine can do the repetative work 24/7 at the same level of performance.

This program will help you! It watches your screen for the trigger on pharmacy software and helps verify that the information entered matches the source document. It places colored boxes over fields to show you if they match (green) or don't match (red). And in YOLO (You only live once!) mode it can even automatically send a key press (F12 by default for PRx) to advance to the next prescription when all fields are green!

Verification of patient DOB, patient address, prescriber address, phone number are option areas can be added. Currently the overall speed is heavily limited by the computer CPU, I'm only keeping the most essential ones mandatory here. Autopilot still needs driver's attention, so use at your own discretion.  


## How It Works

This program monitors the selected screen area for the keyword trigger. Once detects, it will performs the following tasks:
1. Reads the data entered and the source: patient name, prescriber name, drug name, directions/sig, and other optional info from the selected screen regions
2. Compares the entered data against the source document. This is desgined for standard eRx format where the source info are in the fixed locations. (paper Rx pleae check the "future plan" section).
3. Gives a matching score for each field based on the similarity of the entered data and the source document (in a very complicated way, I used fuzzy compare, tokenize, cleaning for titles, middle names, handling abbrevations, etc.) OR AI.
4. Displays colored boxes over the fields to indicate matches (green) or mismatches (red), the passing rate is cutomizable by the user
5. Optionally, a YOLO (you only live once) mode will automatically send a key press (F12 by default) to autopilot the process
6. It polls the screen and read Rx number to check if new Rx is displayed, and will start the process again.

## Fancy skill to make it "smart"
### 🧹 Smart Text Cleaning & Normalization

- **Name Format Normalization**: Automatically converts between "Last, First" and "First Last" formats for accurate patient name comparison
- **Patient Middle Name Handling**: Intelligently matches names with or without middle names/initials (e.g., "John M Smith" matches "John Smith")
- **Prescriber Title Cleaning**: Removes professional titles and suffixes (Dr., MD, PharmD, etc.) from prescriber names for consistent comparison
- **Smart Text Cleaning**: Removes punctuation, normalizes spacing, and handles case differences for consistent comparisons
- **AI-Powered Semantic Matching**: Uses advanced AI models to understand context and meaning, improving accuracy for complex fields like drug names and directions. This system is designed to work with OpenAI-compatible APIs, allowing you to choose between traditional fuzzy matching and AI-based verification on a per-field basis for maximum flexibility and HIPAA compliance. Basic system prompt and user prompt has been provided for easy customization (tested on Gemini and Phi3-mini).

### 💊 Pharmacy-Specific Intelligence

- **Comprehensive Sig Abbreviation Expansion**: Automatically expands common pharmacy abbreviations:
  - **Frequency**: bid → twice daily, tid → three times daily, qid → four times daily
  - **Route**: po → by mouth, sl → sublingual, IM → intramuscular  
  - **Timing**: ac → before meals, pc → after meals, hs → at bedtime
  - **Quantity**: q4h → every 4 hours, prn → as needed
  - **And many more** - handles dozens of standard pharmacy abbreviations
- **Enhanced Drug Name Matching**: Advanced semantic matching for drug names with:
  - **Release formulations**: ER (Extended Release), DR (Delayed Release), XR, LA, SA, SR, CR, etc.
  - **Dosage forms**: TAB (Tablet), CAP (Capsule), TBEC (Tablet Delayed Release), etc.
  - **Administration routes**: PO (By Mouth), IV (Intravenous), IM (Intramuscular), etc.
  - **Common pharmacy terms**: DISP (Dispense), SIG (Directions), PRN (As Needed), etc.
  - **Brand name patterns**: HCL (Hydrochloride), HCTZ (Hydrochlorothiazide), etc.
  - **Multiple matching strategies**: Direct fuzzy matching, token sort ratio, partial matching, and core drug name matching
  - **Easy maintenance**: Add custom abbreviations through external JSON configuration file
- **Flexible Sig Matching**: Accounts for different ways pharmacies express the same directions
- **Route Normalization**: Handles variations in administration routes (oral/po/by mouth)

#### Drug Name Abbreviation Management

The system includes an extensive pharmaceutical abbreviation system that's completely externalized for easy maintenance:

**Easy Updates Method:**
1. Edit `abbreviations.json` in the same folder as the main script
2. Add your custom abbreviations in this format:
```json
{
    " er ": " extended release ",
    " dr ": " delayed release ",
    " tbec ": " tablet delayed release ",
    " your_custom_abbr ": " your expansion "
}
```

**Important formatting rules:**
- Include spaces around abbreviations (e.g., `" er "` not `"er"`)
- Use lowercase for both abbreviations and expansions
- The spaces ensure word boundary matching

**Advanced Drug Name Matching Features:** (this is the most time comsuming part of the program)
- **Preserves dosage information** (5mg, 10ml, etc.) as it's critical for drug identification - metformin 500mg ≠ metformin 1000mg
- **Strict dosage validation**: Different dosages are treated as different medications to prevent dispensing errors
- **Smart salt form handling**: 
  - Normalizes equivalent forms (hydrochloride ↔ hcl) for better matching
  - Maintains distinct salts as separate drugs (metoprolol tartrate ≠ metoprolol succinate)
  - Distinguishes release formulations (IR ≠ ER ≠ SR ≠ XR)
- **Enhanced matching strategies with weighted prioritization**:
  - **Primary drug name matching** (first word gets highest weight for pharmaceutical accuracy)
  - Direct fuzzy matching (weighted higher for exact matches)
  - Token sort ratio (handles word order differences)
  - Token set ratio (handles extra words, weighted lower to reduce false positives)
  - Partial ratio (handles subsets, weighted lower)
  - **Dosage mismatch detection**: Automatically flags when dosages don't match
  - **Salt form validation**: Identifies incompatible salt combinations

**Testing Your Changes:**
- Restart the application to load new abbreviations - Stop then start screen
- Check console output for confirmation that custom abbreviations were loaded
- Use the OCR testing features in the GUI to verify improvements

---

## 🤖 Vision Language Model (VLM) Integration

### Overview

The Vision Language Model integration represents a breakthrough in prescription verification technology. Instead of relying on OCR text extraction, VLM uses AI vision capabilities to directly analyze prescription screenshots, providing superior accuracy and maintaining visual context.

### 🎯 VLM Advantages

**🔍 Direct Image Analysis**
- No OCR text extraction required - AI "sees" the images directly
- Maintains visual context, layout, and formatting information
- Superior handling of handwritten prescriptions and complex layouts
- Reduces errors from OCR misreads and text extraction issues

**🧠 AI-Powered Intelligence**
- Uses vision-capable language models (LLaVA, Qwen2-VL, Phi-3.5-Vision)
- Understands medical terminology and pharmaceutical abbreviations contextually
- Provides confidence scores 0-100 for each prescription field
- Semantic understanding rather than just string matching

**⚙️ Flexible Configuration**
- Works alongside existing OCR system - switch modes easily
- Configurable screenshot regions with visual selection tool
- Customizable prompts and model settings
- Local deployment for HIPAA compliance

### 🚀 VLM Quick Setup

#### 1. Install VLM Dependencies
```bash
pip install openai pillow pyautogui  # Already included in requirements.txt
```

#### 2. Set Up VLM Server
Choose your preferred local VLM server:

**Recommended: Ollama (Easiest)**
```bash
# Install Ollama from https://ollama.ai
ollama pull llava-next   # or qwen2-vl, phi3.5-vision
ollama serve
```

**Alternative: LlamaFile**
```bash
# Download vision model from https://huggingface.co
# Run with multimodal support:
./llamafile --server --multimodal --port 8081
```

#### 3. Configure VLM in Streamlit
1. Launch Streamlit: `streamlit run streamlit_app.py`
2. Navigate to **"VLM Configuration"** in sidebar
3. **Model Settings Tab**:
   - Set Base URL (e.g., `http://localhost:11434/v1` for Ollama)
   - Enter your model name (e.g., `llava-next`)
   - Configure generation settings

#### 4. Set Up Screenshot Regions
**Visual Method (Recommended)**:
1. In VLM Configuration → **Region Setup Tab**
2. Click **"🎯 Launch Coordinate Helper"**
3. VLM GUI opens with visual selection:
   - Take screenshot of your pharmacy software
   - Click and drag to select **Data Entry Region** (where data is entered)
   - Click and drag to select **Source Region** (original prescription area)
   - Save configuration

**Manual Method**:
- Enter coordinates manually in the Region Setup tab
- Use "Test Capture" buttons to verify regions

#### 5. Test Your Setup
1. Go to **Testing Tab** in VLM Configuration
2. Click **"🔍 Test VLM Connection"** - should show ✅ success
3. Click **"🚀 Run Complete VLM Verification"** to test full workflow

#### 6. Switch to VLM Mode
1. Go to **Monitor & Logs** page
2. Change **Verification Method** from "OCR" to "Vision Language Model"
3. Start monitoring - VLM will now handle all verifications!

### 🛠️ VLM Configuration GUI

**Launch VLM GUI**: `python vlm_gui.py` or use the Streamlit button

**Features**:
- 📸 **Visual Screenshot Capture**: See your actual pharmacy software
- 🖱️ **Click & Drag Selection**: Visually define regions instead of typing coordinates
- 🔴🔵 **Color-Coded Regions**: Red for data entry, blue for source prescription
- 📊 **Live Coordinates**: Real-time pixel coordinate display
- 💾 **Auto-Save**: Direct integration with `vlm_config.json`
- 🧪 **Built-in Testing**: Test VLM connection and capture regions

### 📋 VLM vs OCR Comparison

| Feature | Traditional OCR | VLM Integration |
|---------|----------------|-----------------|
| **Accuracy** | Good for typed text | Superior for all text types |
| **Handwriting** | Poor | Excellent |
| **Setup Complexity** | Requires precise coordinates | Simple region selection |
| **Visual Context** | Lost during text extraction | Preserved in analysis |
| **Error Types** | OCR misreads | Rare AI hallucination |
| **Speed** | Fast | Moderate (model-dependent) |
| **Resource Usage** | CPU only | Benefits from GPU |
| **Flexibility** | Fixed string matching | Semantic understanding |

### 🔧 Supported VLM Models

**Recommended Models** (tested):
- **LLaVA-Next**: Excellent general vision capabilities
- **Qwen2.5-VL**: Superior text recognition, multilingual
- **Phi-3.5-Vision**: Microsoft's efficient model, good balance

**Model Selection Tips**:
- **Q4_K_M quantization**: Best balance of accuracy vs speed
- **7B parameters**: Optimal for prescription verification
- **Local deployment**: Use Ollama/LlamaFile for HIPAA compliance

### 🧪 VLM Testing & Validation

**Built-in Test Script**:
```bash
python test_vlm.py
```

**Test Checklist**:
- ✅ Dependencies installed
- ✅ VLM configuration valid
- ✅ Model connection successful
- ✅ Screenshot regions working
- ✅ Integration test passed

**Troubleshooting VLM Issues**:

| Problem | Solution |
|---------|----------|
| "VLM connection failed" | Verify server running, check base_url |
| "Image input not supported" | Ensure model has vision capabilities |
| "No scores returned" | Check prompts, verify image contains text |
| "Screenshot capture failed" | Validate region coordinates |

### 📈 VLM Performance Optimization

**Speed Optimization**:
- Use quantized models (Q4_K_M recommended)
- Optimize screenshot region sizes
- Enable GPU acceleration if available
- Disable auto-enhance for clear images

**Accuracy Tuning**:
- Customize system prompts for your pharmacy workflow
- Adjust confidence thresholds per field type
- Use pharmacy-specific terminology in prompts
- Fine-tune regions to capture only relevant text

### 🔒 VLM HIPAA Compliance

**Local Deployment Benefits**:
- Patient data never leaves your premises
- Full control over AI model and processing
- No internet connection required for verification
- Complete audit trail of all processing

**Recommended Local Setup**:
- Use Ollama or LlamaFile for easy local deployment
- Deploy on dedicated pharmacy server for best performance
- Configure firewall to block external model access
- Regular backups of model and configuration files

---

## Setup Instructions

### Step 1: Download the Project Files

1.  Click the green **`< > Code`** button on the top-right of the GitHub page.
2.  Select **`Download ZIP`** from the dropdown menu.
3.  Find the downloaded ZIP file (usually in your `Downloads` folder) and right-click it.
4.  Select **`Extract All...`** and choose a location to save the files (like your Desktop).

### Step 2: Install Python

If you don't have Python, you'll need to install it.

1.  Go to the official Python download page: [https://www.python.org/downloads/](https://www.python.org/downloads/)
2.  Download the latest version for Windows.
3.  Run the installer. **Crucially, check the box at the bottom that says "Add Python to PATH"** before clicking `Install Now`. This makes the next steps much easier.


### Step 3: Install Required Libraries

1.  Open the **Command Prompt** (Windows) or **Terminal** (macOS/Linux). You can find it by clicking the Start Menu and typing `cmd` (Windows) or searching for Terminal (macOS/Linux).
2.  Navigate to the project folder you extracted in Step 1. Type `cd` followed by the path to the folder. For example, if it's on your desktop, you would type:
    ```
    cd C:\Users\YourUsername\Desktop\HayatPrecheck
    ```
3.  Once you are in the correct folder, copy and paste the following command into the Command Prompt and press **Enter**:
    ```
    pip install -r requirements.txt
    ```
4.  Advance user please create a virtual environment to avoid conflicts with other Python projects:
    ```
    python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    ```

### Step 4: OCR Engine Setup

🚀 **Automatic OCR Selection** - The system now features **intelligent OCR provider selection** with GPU detection for optimal performance!

#### Automatic OCR Provider Selection

The system automatically selects the best OCR provider based on your hardware capabilities:

**🎯 Smart Selection Algorithm:**
1. **EasyOCR + nVIDIA GPU** → Best accuracy (90%+) and speed with CUDA-compatible GPU 
2. **Tesseract** → Best speed when no GPU available
3. **EasyOCR CPU** → Fallback option for compatibility, slower on CPU but more accurate

**🔧 Key Features:**
- **GPU Detection**: Automatically detects CUDA-compatible GPUs using PyTorch
- **Zero Configuration**: Works out-of-the-box with optimal settings
- **Graceful Fallback**: Falls back intelligently when hardware changes
- **Performance Optimization**: Always selects the best provider for your system
- **Clear Logging**: Detailed logs explain selection decisions


#### Quick OCR Setup

**One-Command Setup (Recommended):**
```bash
python launcher.py
```
The launcher will:
- ✅ **Auto-detect** GPU availability and OCR providers
- ✅ **Install EasyOCR** automatically if needed
- ✅ **Configure** your system optimally with AUTO mode
- ✅ **Test** the installation
- ✅ **Set up intelligent selection** based on your hardware

#### Manual Installation Options

**EasyOCR (Auto-installed when needed):**
```bash
pip install easyocr
```

**Tesseract (Alternative for CPU-only systems):**
```bash
pip install pytesseract
```

#### Tesseract Installation

1.  Download the **Tesseract OCR** for your operating system:
    - **Windows**: [Download Tesseract for Windows](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe)
    - **macOS**: `brew install tesseract` (requires [Homebrew](https://brew.sh/))
    - **Linux**: `sudo apt-get install tesseract-ocr`
2.  Run the installer and follow the installation wizard with Admin credentials
3.  The system will automatically detect and use Tesseract when appropriate

## 📱 Features

### Coordinate Setup GUI
- **📸 Screenshot capture** with one click
- **🖱️ Click and drag** region selection
- **🔍 Live OCR testing** to verify text detection
- **📏 Visual overlays** showing selected regions
- **💾 Automatic saving** of coordinates
- **✅ Validation** to ensure setup is correct

### Web Monitoring Dashboard
- **🟢/🔴 Status indicator** (Running/Stopped)
- **📊 Real-time charts** of verification results
- **📋 Live log display** with filtering
- **📈 Performance metrics** and accuracy tracking
- **🎯 Field-by-field analysis** (which fields are problematic)
- **📱 Mobile responsive** - monitor from anywhere

## 🔧 Configuration

**🎯 Setup Notes:**
- Works with any screen resolution and window configuration
- Use the GUI coordinate tools to configure regions for your setup

## How to Stop the Program

To stop the program, go to the Command Prompt window where it is running and press the **`Ctrl`** + **`C`** keys at the same time. Or simply close the Command Prompt window.

---

## Coordinate Adjustment Tools

### Enhanced GUI Tool (Recommended)

**File:** `settings_gui.py`

A comprehensive graphical tool with advanced features for both coordinate adjustment and general settings:

```bash
python settings_gui.py
```

**Features:**
- **Visual Interface**: See your desktop screenshot with overlay rectangles
- **Drag & Drop**: Click and drag to define regions visually
- **Real-time Preview**: See coordinate changes immediately
- **Zoom Controls**: Zoom in/out for precise adjustment (View menu)
- **OCR Testing**: Test selected regions to verify text detection
- **Configuration Management**: Import/export settings, manual backups available
- **Validation**: Check all regions for potential issues
- **Auto-Save**: Optionally save changes automatically
- **Keyboard Shortcuts**: Ctrl+S to save, F5 for new screenshot
- **General Settings**: Configure matching thresholds and automation options

**Usage:**
1. Run the tool and take a screenshot of your pharmacy software
2. Select "trigger" first and drag around the Trigger text
3. For each field, select the field name and region type
4. Drag rectangles around the appropriate text areas
5. Use "Test OCR" to verify the selection captures text correctly
6. Adjust general settings like matching thresholds and automation options
7. Save configuration when complete

### Matching Thresholds

Control how strict the text matching is for each field type:

- **Patient Name Threshold** (default: 75%): How similar patient names must be to consider a match
- **Prescriber Name Threshold** (default: 75%): How similar prescriber names must be to consider a match  
- **Drug Name Threshold** (default: 65%): How similar drug names must be to consider a match
- **Directions/Sig Threshold** (default: 65%): How similar directions must be to consider a match

**Higher values** = More strict matching (fewer false positives, but might miss valid matches)
**Lower values** = More lenient matching (might catch more matches, but could have false positives)

### Automation Settings

Configure what happens when all fields match:

- **Enable/Disable Automation**: Whether to automatically send a key when all fields are green
- **Key to Send**: Which key to press (F1-F12, Enter, Tab, Space, Escape)
- **Delay**: How long to wait (in seconds) before sending the key

**Example Use Cases:**
- Set key to "F12" to automatically advance to the next prescription
- Set key to "Enter" to automatically submit the form
- Set delay to 1.0s to give time to visually confirm before advancing

---

## 🌐 Streamlit Web Interface - VISUALLY Friendly

### Quick Start with Streamlit

#### Option 1: Auto-Launcher (Recommended)
- **Windows**: Run `start.bat` to launch the Streamlit app
- **Any OS**: Run `python launcher.py` and choose option 3

#### Option 2: Manual Launch
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Features

- 🏠 **Single Interface**: Everything in one web application
- 📊 **Real-time Monitoring**: Live charts and analytics of verification performance  
- ⚙️ **Visual Settings**: Easy configuration with sliders and visual feedback
- 📱 **Mobile Friendly**: Access from any device on your network
- 📈 **Advanced Analytics**: Time-based filtering and performance metrics
- 🔧 **Coordinate Setup**: Screenshot-assisted coordinate configuration
- 💾 **Backup**: Allow configuration backups for different monitors
- 🎯 **OCR Testing**: Test regions directly in the interface


### Access Your Application
Once launched, open your web browser to: http://localhost:8501

---

## 🆘 Troubleshooting

### "tesseract is not installed or it's not in your PATH" Error

⚠️ **Note**: This error should be rare now that the system uses **automatic OCR selection**. If you encounter this, the system is trying to use Tesseract but it's not properly installed.

**Quick Fix**: Ensure your `config.json` has `"ocr_provider": "auto"` to use intelligent OCR selection.

If you specifically need Tesseract, try these solutions:

**Option 1: Add Tesseract to your system PATH**
1. Find where Tesseract was installed (usually `C:\Program Files\Tesseract-OCR`). Please make sure you installed it with admin credentials and selected "install for all users".
2. Add this path to your Windows PATH environment variable:
   - Press `Windows + R`, type `sysdm.cpl`, and press Enter
   - Click "Environment Variables"
   - Under "System Variables", find and select "Path", then click "Edit"
   - Click "New" and add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR`)
   - Click "OK" on all windows
3. Restart your Command Prompt and try running the program again

**Option 2: Create a local tesseract folder (if Option 1 doesn't work)**
1. In your project folder, create a new folder named `tesseract`
2. Copy `tesseract.exe` and the `tessdata` folder from your Tesseract installation directory to this new folder
3. Your project should then look like this:
   ```
   HayatPrecheck-main/
   |-- 📁 config/                     # Configuration files
   |   |-- config.json                # Main configuration file
   |   |-- vlm_config.json           # VLM-specific configuration
   |   |-- abbreviations.json        # Pharmacy abbreviations database
   |   |-- 1080laptop.json          # Sample coordinate config (1080p)
   |   |-- 4kpc.json                # Sample coordinate config (4K)
   |-- 📁 core/                       # Core verification engine
   |   |-- verification_controller.py # Main verification engine
   |   |-- comparison_engine.py      # Core field comparison logic
   |   |-- ocr_provider.py          # OCR engine management
   |   |-- settings_manager.py      # Configuration management
   |   |-- logger_config.py         # Logging configuration
   |-- 📁 ai/                         # AI and machine learning modules
   |   |-- ai_verifier.py           # AI-powered verification module
   |   |-- vlm_verifier.py          # VLM verification engine
   |-- 📁 ui/                         # User interface modules
   |   |-- streamlit_app.py         # Web monitoring dashboard
   |   |-- streamlit_ai_page.py     # AI settings interface
   |   |-- streamlit_vlm_page.py    # VLM configuration interface
   |   |-- settings_gui.py          # GUI coordinate setup tool
   |   |-- vlm_gui.py               # VLM visual coordinate selection GUI
   |-- 📁 tools/                      # Testing and utility tools
   |   |-- test_vlm.py              # VLM testing and validation script
   |-- 📁 scripts/                    # Launcher scripts
   |-- 📁 config_backups/            # Automatic configuration backups
   |-- 📁 __pycache__/               # Python cache directory
   |-- launcher.py                   # Main launcher with setup wizard
   |-- start.bat                     # Windows batch launcher
   |-- requirements.txt              # Python dependencies
   |-- README.md                     # This documentation
   |-- VLM_README.md                 # VLM-specific documentation (legacy)
   |-- AGENTS.md                     # Development guidelines
   |-- venv/ or .venv/              # Python virtual environment (optional)
   ```


### "Web interface won't open"
- Try: `python -m streamlit run streamlit_app.py`
- Check firewall isn't blocking port 8501
- Try `http://127.0.0.1:8501` or `http://localhost:8501` instead
- Streamlit runs on http, not https by default. Please ignore all safety warnings from your browser.
- Try another browser or incognito mode

### "OCR not working"

🚀 **With Automatic OCR Selection**: OCR issues should be minimal! The system now automatically selects and configures the best OCR engine for your hardware.


**If you're having OCR issues:**

- **Adjust Coordinates**: Use the GUI tool to ensure regions are set correctly, avoid artifacts, boxes, lines, or other distractions. Make it as small as possible.
- **Test OCR Regions**: Use the "Test OCR" feature in the GUI to verify each region captures text correctly
- **Test different OCR providers**: I've noticed different performance on different hardware/computer/user behavior(??). Test and pick what works best. 

**Automatic Fallback System:**
- It will try to read again if infomation is empty or not detected, retry times and interval can be configured.
- A delay can be set to allow the system to wait for the screen to fully load before reading


---

## 🔒 Privacy & Security

- ✅ **Completely local** - no data sent anywhere
- ✅ **HIPAA friendly** - patient data stays on your machine  
- ✅ **No internet required** for core functionality
- ✅ **Open source** - you can see exactly what it does
- ✅ **Screenshots stay on your machine**
- ✅ **Your data never leaves your computer**

---

## 🎉 Success Tips

1. **Take clear screenshots** - make sure text is visible
2. **Test OCR immediately** after selecting regions
3. **Start with trigger region** - get this working first
4. **Check validation** before running monitoring
5. **Use the smallest area** based on your software, avoid box, line, or any other artifact that may interfere with the OCR
6. **Adjust matching thresholds** based on your data entry habits, pharamcy may have different sig translation
7. **Monitor logs** to fine-tune accuracy
8. **Prompt Engineering** for better AI response

---

## 📈 Future Improvements - My favorite part

### ✅ Completed: Vision Language Model Integration

**🎉 Major Milestone Achieved!** The VLM integration is now fully operational, representing a significant leap forward in prescription verification technology:

- ✅ **Direct Image Analysis**: No more OCR extraction - AI sees images directly
- ✅ **Visual GUI Configuration**: Point-and-click region selection tool
- ✅ **Multiple Model Support**: LLaVA, Qwen2-VL, Phi-3.5-Vision compatibility
- ✅ **Local Deployment**: Complete HIPAA compliance with on-premises processing
- ✅ **Seamless Integration**: Easy switching between OCR and VLM modes
- ✅ **Superior Handwriting Support**: Handles complex layouts and handwritten prescriptions

### 🚀 Next Phase: Advanced Architecture Evolution

#### Phase 1: Enhanced VLM Capabilities (In Progress)
- **Multi-Model Ensemble**: Combine multiple VLM models for consensus-based verification
- **Custom Prompt Engineering**: Pharmacy-specific prompt templates for better accuracy
- **Adaptive Region Detection**: Automatic region detection using computer vision
- **Confidence Calibration**: Fine-tuned confidence scoring based on real-world data

#### Phase 2: Centralized Processing Hub
Building on the successful VLM foundation, the next evolution involves centralized architecture:

**Central Processing Hub Vision:**
- **One Powerful Machine**: A dedicated high-performance computer with enterprise-grade GPU serving multiple pharmacy locations
- **VLM-Powered Backend**: Centralized VLM processing with specialized pharmacy models
- **Network-Based Service**: All pharmacies connect to the central hub for processing
- **Scalable Architecture**: One central system handling dozens of pharmacy workstations simultaneously

#### Phase 3: Advanced AI Integration
- **Custom VLM Training**: Fine-tune models specifically on pharmacy data
- **Semantic Drug Matching**: Replace string matching with AI-powered semantic understanding
- **Contextual Analysis**: Full prescription context analysis beyond individual fields
- **Predictive Error Detection**: AI predicts likely errors before they occur

#### Phase 4: Enterprise Intelligence
- **Multi-Pharmacy Analytics**: Aggregated insights across pharmacy networks
- **Pattern Recognition**: Identify common error patterns and improvement opportunities
- **Automated Quality Assurance**: Continuous model improvement based on verification outcomes
- **Regulatory Compliance**: Enhanced audit trails and compliance reporting

**Why This Centralized Approach:**
- **Cost Efficiency**: One powerful VLM server replaces dozens of individual installations
- **Performance**: Enterprise-grade GPU processing delivers consistent high-speed verification
- **Model Management**: Centralized model updates and improvements
- **Scalability**: Easy addition of new pharmacy locations
- **Specialized Models**: Deploy pharmacy-specific VLM models optimized for prescription verification

### 🌟 Current Achievement Impact

The VLM integration represents a paradigm shift from traditional OCR-based systems:

**Before VLM**: Text extraction → String matching → Verification
**After VLM**: Image → AI Vision Analysis → Semantic Verification

This advancement opens possibilities for:
- **Handwritten Prescription Support**: Previously impossible with OCR
- **Layout-Independent Processing**: Works with any prescription format
- **Contextual Understanding**: AI understands medication context and relationships
- **Reduced Setup Complexity**: Visual region selection instead of precise coordinate tuning

*The VLM integration proves that AI-powered pharmacy automation is not just feasible but highly effective. This foundation enables the next phase of centralized, enterprise-scale prescription verification systems.*

---

## 📁 Project Organization

### Organized Folder Structure

The project has been reorganized into logical modules for better maintainability and easier development:

```
📁 HayatPrecheck/
├── 🚀 launcher.py              # Main entry point - start here!
├── 📄 requirements.txt         # Dependencies
├── 📖 README.md               # This documentation
├── 📁 config/                 # All configuration files
├── 📁 core/                   # Core verification engine
├── 📁 ai/                     # AI and VLM modules  
├── 📁 ui/                     # User interfaces (Streamlit + GUI)
├── 📁 tools/                  # Testing and utilities
└── 📁 scripts/                # Launcher scripts
```

### 🎯 Quick Access

**For Users:**
- **Start Here**: `python launcher.py` - Main launcher with setup wizard
- **Web Interface**: `streamlit run ui/streamlit_app.py` - Browser-based control
- **VLM Setup**: `python ui/vlm_gui.py` - Visual coordinate selection

**For Developers:**
- **Core Engine**: `core/` - Main verification logic
- **AI Integration**: `ai/` - AI and VLM components
- **Configuration**: `config/` - All settings and coordinate files

### 🔧 Module Responsibilities

| Module | Purpose | Key Files |
|--------|---------|-----------|
| **config/** | Settings & coordinates | config.json, vlm_config.json, abbreviations.json |
| **core/** | Verification engine | verification_controller.py, comparison_engine.py |
| **ai/** | AI & vision models | ai_verifier.py, vlm_verifier.py |
| **ui/** | User interfaces | streamlit_app.py, settings_gui.py, vlm_gui.py |
| **tools/** | Testing & utilities | test_vlm.py |

This organization makes the codebase more maintainable, easier to navigate, and better prepared for future enhancements.

---
