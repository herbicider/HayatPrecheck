# What's new?

**08/19/2025: AI incorporation**

**AI-Powered Verification**: The system now is able to integrate with OpenAI-compatible APIs for intelligent text comparison and semantic matching. Configure your preferred AI endpoint, API key, and model through the dedicated AI settings page. Customize system and user prompts to optimize verification accuracy for your specific workflow. It allows you to choos what to use AI and what to use traditional fuzzy matching on a per-field basis for maximum flexibility and HIPAA compliance (NEVER send patient info to online AI).

**🔒 HIPAA Compliance Notice**: For maximum privacy protection, use local AI models (like Ollama, LM Studio, or GPT4All) instead of cloud-based APIs when processing patient data. During development and testing, I used a local Phi3-mini model to ensure sensitive information never leaves the premises.

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

This program will help you! It watches your screen for the "Pre-Check Rx" pharmacy software (PioneerRx) and helps verify that the information entered matches the source document. It places colored boxes over fields to show you if they match (green) or don't match (red). And in YOLO (You only live once!) mode it can even automatically send a key press (F12 by default for PRx) to advance to the next prescription when all fields are green!

This app is built for the PioneeRx pharmacy dispensing software, if you use a differnt software and need help adapting this app to your setup, please feel free to reach out or simply customize to your keyword trigger in the codebase (verification_controller.py file)! The concept of precheck verification for data entry accuracy shall be the same or very similar regardless of the software you use.

And yes, verification of patient DOB, patient address, prescriber address, phone number, fax number etc... can be added easily, currently the overall speed is heavily limited by the computer CPU, I'm only keeping the most essential ones here. Autopilot still needs driver's attention, so use at your own discretion.  


## How It Works

This program monitors the selected screen area for the "Pre-Check Rx" keyword trigger. Once detects, it will performs the following tasks:
1. Reads the data entered and the source: patient name, prescriber name, drug name, and directions/sig from the selected screen regions
2. Compares the entered data against the source document. This is desgined for standard eRx format where the source info are in the fixed locations. (paper Rx pleae check the "future plan" section) 
3. Gives a matching score for each field based on the similarity of the entered data and the source document (in a very complicated way, I used fuzzy compare, tokenize, cleaning for titles, middle names, handling abbrevations, etc.)
4. Displays colored boxes over the fields to indicate matches (green) or mismatches (red), the passing rate is cutomizable by the user
5. Optionally, a YOLO (you only live once) mode will automatically send a key press (F12 by default) to autopilot the process
6. It polls the screen and read Rx number to check if new Rx is displayed, and will start the process again.

## Fancy skill to make it "smart"
### 🧹 Smart Text Cleaning & Normalization

- **Name Format Normalization**: Automatically converts between "Last, First" and "First Last" formats for accurate patient name comparison
- **Patient Middle Name Handling**: Intelligently matches names with or without middle names/initials (e.g., "John M Smith" matches "John Smith")
- **Prescriber Title Cleaning**: Removes professional titles and suffixes (Dr., MD, PharmD, etc.) from prescriber names for consistent comparison
- **Smart Text Cleaning**: Removes punctuation, normalizes spacing, and handles case differences for consistent comparisons

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

**All abbreviations are now managed externally** - no code editing required! The system loads all pharmaceutical abbreviations from `abbreviations.json` at startup, including:
- Release formulations (ER, DR, XR, LA, etc.)
- Dosage forms (TAB, CAP, TBEC, etc.)  
- Administration routes (PO, IV, IM, etc.)
- Frequency terms (BID, TID, QID, etc.)
- Common pharmacy abbreviations and brand names

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
- Restart the application to load new abbreviations
- Check console output for confirmation that custom abbreviations were loaded
- Use the OCR testing features in the GUI to verify improvements

### 🎯 Advanced Matching Logic

- **Flexible Matching**: Uses a 75%+ similarity threshold to account for minor OCR differences and typos
- **Context-Aware Processing**: Different cleaning rules for names vs. drug names vs. directions
- **Adaptive Timing**: Responds quickly to screen changes (0.1s) and slows down when screen is static to save CPU
- **Prescription Change Detection**: Automatically detects when you move to a new prescription and re-runs verification

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

![Add Python to PATH](https://docs.python.org/3/_images/win_installer.png)

### Step 3: Install Required Libraries

1.  Open the **Command Prompt** (Windows) or **Terminal** (macOS/Linux). You can find it by clicking the Start Menu and typing `cmd` (Windows) or searching for Terminal (macOS/Linux).
2.  Navigate to the project folder you extracted in Step 1. Type `cd` followed by the path to the folder. For example, if it's on your desktop, you would type:
    ```
    cd C:\Users\YourUsername\Desktop\HayatPrecheck-main
    ```
3.  Once you are in the correct folder, copy and paste the following command into the Command Prompt and press **Enter**:
    ```
    pip install -r requirements.txt
    ```

### Step 4: OCR Engine Setup

🚀 **Automatic OCR Selection** - The system now features **intelligent OCR provider selection** with GPU detection for optimal performance!

#### Automatic OCR Provider Selection

The system automatically selects the best OCR provider based on your hardware capabilities:

**🎯 Smart Selection Algorithm:**
1. **EasyOCR + GPU** → Best accuracy (90%+) when CUDA-compatible GPU available
2. **Tesseract** → Best speed when no GPU available
3. **EasyOCR CPU** → Fallback option for compatibility

**🔧 Key Features:**
- **GPU Detection**: Automatically detects CUDA-compatible GPUs using PyTorch
- **Zero Configuration**: Works out-of-the-box with optimal settings
- **Graceful Fallback**: Falls back intelligently when hardware changes
- **Performance Optimization**: Always selects the best provider for your system
- **Clear Logging**: Detailed logs explain selection decisions

#### Available OCR Providers

| OCR Engine | Speed | Accuracy | Setup | Best For | Auto-Selected When |
|------------|-------|----------|--------|----------|---------------------|
| **EasyOCR** ⭐ | Good | 90% | Auto | **GPU systems** | GPU available |
| **Tesseract** | Fast | 85% | Manual | **CPU systems** | No GPU available |

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

#### Tesseract Manual Installation

If you want to use Tesseract OCR specifically (or as a fallback):

1.  Download the **Tesseract OCR** for your operating system:
    - **Windows**: [Download Tesseract for Windows](https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.3.0.20221214.exe)
    - **macOS**: `brew install tesseract` (requires [Homebrew](https://brew.sh/))
    - **Linux**: `sudo apt-get install tesseract-ocr`
2.  Run the installer and follow the installation wizard.
3.  The system will automatically detect and use Tesseract when appropriate

#### OCR Provider Options

**Auto (Default & Recommended):**
- ✅ **Smart GPU detection** - automatically uses GPU if available
- ✅ **Optimal performance** - selects best provider for your system
- ✅ **EasyOCR with GPU** → best accuracy (90%+)
- ✅ **Tesseract fallback** → best speed when no GPU
- ✅ **Zero configuration** needed
- 🎯 Set in config: `"ocr_provider": "auto"`

**EasyOCR:**
- ✅ **Easy installation** - just `pip install easyocr`
- ✅ **High accuracy** on medical text (90%)
- ✅ **No external dependencies** required
- ✅ **GPU acceleration** support (automatic detection)
- ⚠️ **Slower on CPU** compared to Tesseract

**Tesseract:**
- ✅ **Very reliable and stable**
- ✅ **Fast startup** time
- ✅ **CPU-optimized** performance
- ✅ **Wide compatibility**
- ⚠️ **Requires separate binary installation**
- ⚠️ **Lower accuracy** compared to AI-based OCR (85%)

#### Performance Notes

- **EasyOCR in CPU mode** is often faster than GPU mode for small text regions
- **Tesseract** excels at speed and reliability, especially on older hardware
- The system **automatically falls back** to Tesseract if EasyOCR fails
- Provider caching prevents reinitialization for better performance

---

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

🚀 **NEW: Automatic OCR Selection** - Choose the best OCR engine automatically or manually configure for your specific needs!

### OCR Engine Configuration

The system now supports automatic OCR provider selection with intelligent GPU detection:

| OCR Engine | Speed | Accuracy | Setup | Best For | Status |
|------------|-------|----------|--------|----------|--------|
| **Auto** ⭐ | Optimal | 95% | Zero | **Recommended for all users** | ✅ Default |
| **EasyOCR** | Good | 90% | Auto | **GPU systems, high accuracy** | ✅ Available |
| **Tesseract** | Fast | 85% | Manual | **CPU systems, reliability** | ✅ Available |

**Auto-Selection Benefits:**
- ✅ **Zero Configuration**: Works immediately with optimal settings
- ✅ **Hardware-Aware**: Automatically detects and uses GPU when available
- ✅ **Performance Optimized**: Always selects the best provider for your system
- ✅ **Graceful Fallback**: Intelligently switches providers when needed
- ✅ **User-Friendly**: No technical knowledge required

**Manual Configuration Options:**
- **GPU Systems**: Set `"ocr_provider": "easyocr"` for maximum accuracy
- **CPU Systems**: Set `"ocr_provider": "tesseract"` for maximum speed
- **Auto Mode**: Set `"ocr_provider": "auto"` for intelligent selection (recommended)

**Performance Tuning:**
- **GPU Acceleration**: Automatically enabled when compatible GPU detected
- **Confidence Thresholds**: Adjust to filter out low-quality OCR results
- **Fallback Logic**: System automatically switches providers if primary fails

The system automatically:
- ✅ Creates configuration files
- ✅ Validates coordinate setup  
- ✅ Backs up settings
- ✅ Tests OCR functionality
- ✅ Selects optimal OCR engine based on hardware
- ✅ Falls back gracefully when providers fail

## 🚀 Quick Start Guide

**Recommended:** Use the super easy launcher for the best experience!

### Option 1: Auto-Launcher (Easiest)
```bash
python launcher.py
```
**Or double-click:** `start.bat` (Windows)

**Choose option 3** to launch both setup GUI and monitoring interface!

### Option 2: Step-by-Step
1. **Set up coordinates:** `python settings_gui.py`
2. **Start monitoring:** `streamlit run streamlit_app.py`

### Option 3: Advanced Users Only
For direct core engine access:
```bash
python verification_controller.py
```

The app will run and log for debug will continue to show up in the command window.

**🎯 Setup Notes:**
- Works with any screen resolution and window configuration
- Use the GUI coordinate tools to configure regions for your setup
- Your pharmacy software can be windowed or maximized - just adjust coordinates accordingly
- **If colored boxes appear in wrong locations:** Use the coordinate adjustment tools below

## How to Stop the Program

To stop the program, go to the Command Prompt window where it is running and press the **`Ctrl`** + **`C`** keys at the same time. Or simply close the Command Prompt window.

---

## Coordinate Adjustment Tools

**🎯 FLEXIBILITY:** The program works with any screen resolution, window size, or pharmacy software layout. Use these tools to configure the regions for your specific setup - no need to change your monitor resolution or window configuration!

The program includes two tools for adjusting screen coordinates and general settings when the default configuration doesn't work for your setup:

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
2. Select "trigger" first and drag around the "Pre-Check Rx" text
3. For each field, select the field name and region type
4. Drag rectangles around the appropriate text areas
5. Use "Test OCR" to verify the selection captures text correctly
6. Adjust general settings like matching thresholds and automation options
7. Save configuration when complete

**Requirements:** 
- tkinter (usually included with Python)


### Command-Line Tool (Fallback) 
This was created for debugging purpose in the begining, may not update in the future as the program is stablized and GUI is more user-friendly.

**File:** `settings_cli.py`

A simple command-line tool for when the GUI is not available:

```bash
# View current coordinates
python settings_cli.py show

# Edit specific field coordinates
python settings_cli.py edit patient_name entered

# Validate all coordinates
python settings_cli.py validate

# Create backup (manual - when needed)
python settings_cli.py backup

# View current settings
python settings_cli.py settings show

# Adjust matching thresholds
python settings_cli.py settings set threshold patient 70
python settings_cli.py settings set threshold drug 80

# Configure automation
python settings_cli.py settings set automation enable
python settings_cli.py settings set automation key f11
python settings_cli.py settings set automation delay 1.0
```

**Features:**
- Works in any terminal environment
- View and edit coordinates manually
- View and modify general settings (thresholds, automation)
- Validation and backup functions
- No GUI dependencies required

---

## General Settings Configuration

Both the GUI and command-line tools now allow you to configure general program settings:

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

**🆕 NEW!** I now offer a modern web-based interface that combines monitoring and configuration in one easy-to-use application. I may not prioritize this side as I'm leaning toward more on the tech part. 

### Quick Start with Streamlit

#### Option 1: Auto-Launcher (Recommended)
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
   |-- tesseract/                    # Optional Tesseract installation
   |   |-- tesseract.exe
   |   |-- tessdata/
   |-- ai_verifier.py               # AI-powered verification module
   |-- comparison_engine.py         # Core field comparison logic
   |-- launcher.py                  # Main launcher with setup wizard
   |-- ocr_provider.py             # OCR engine management
   |-- settings_gui.py             # GUI coordinate setup tool
   |-- settings_cli.py             # Command-line setup tool
   |-- settings_manager.py         # Configuration management
   |-- streamlit_app.py            # Web monitoring dashboard
   |-- streamlit_ai_page.py        # AI settings interface
   |-- verification_controller.py   # Main verification engine
   |-- logger_config.py            # Logging configuration
   |-- start.bat                   # Windows batch launcher
   |-- config.json                 # Main configuration file
   |-- abbreviations.json          # Pharmacy abbreviations database
   |-- requirements.txt            # Python dependencies
   |-- README.md                   # This documentation
   |-- AGENTS.md                   # Development guidelines
   |-- 1080laptop.json            # Sample coordinate config (1080p)
   |-- 4kpc.json                  # Sample coordinate config (4K)
   |-- __pycache__/               # Python cache directory
   |-- config_backups/            # Automatic configuration backups
   |-- venv/ or .venv/            # Python virtual environment (optional)
   ```

### "GUI won't start"
- Make sure you have Python with tkinter: `python -c "import tkinter"`
- On some Linux systems: `sudo apt-get install python3-tk`

### "Web interface won't open"
- Try: `python -m streamlit run streamlit_app.py`
- Check firewall isn't blocking port 8501
- Try `http://127.0.0.1:8501` or `http://localhost:8501` instead
- Streamlit runs on http, not https by default. Please ignore all safety warnings from your browser.

### "OCR not working"

🚀 **With Automatic OCR Selection**: OCR issues should be minimal! The system now automatically selects and configures the best OCR engine for your hardware.


**If you're having OCR issues:**

**For Auto Mode (Recommended):**
- Check that `"ocr_provider": "auto"` is set in your `config.json`
- The system will automatically select and configure the best available provider
- Check console output for auto-selection decisions and any error messages

**For EasyOCR:**
- Check that `"ocr_provider": "easyocr"` is set in your `config.json`
- Verify EasyOCR installed correctly: Run `python -c "import easyocr; print('EasyOCR working!')"`
- For GPU acceleration, ensure you have a compatible NVIDIA GPU and CUDA installed

**For Tesseract:**
- Check that `"ocr_provider": "tesseract"` is set in your `config.json`
- Install Tesseract: Download from [GitHub releases](https://github.com/tesseract-ocr/tesseract)
- Windows: Add to PATH or put in project folder
- Verify installation: Run `tesseract --version`

**Automatic Fallback System:**
- If your selected OCR provider fails, the system automatically falls back to available alternatives
- Check the console output for fallback messages
- The system will continue working even if your preferred provider has issues

**Performance Comparison:**
- **Auto Mode**: Automatically selects optimal provider (recommended)
- **EasyOCR**: Good balance of speed and accuracy (90% accuracy)
- **Tesseract**: Fast initialization, reliable fallback (85% accuracy)

## 🚀 Performance Improvements with Automatic OCR Selection

**Significant Speed and Accuracy Improvements with Intelligent Provider Selection:**

| OCR Provider | Processing Time | Accuracy | Best Use Case | Auto-Selected When |
|--------------|----------------|----------|---------------|-------------------|
| **Auto Mode** | **Optimal** | **Up to 90%** | **Recommended for all users** | **Always available** |
| **EasyOCR** | ~600ms | 90% | **GPU systems, high accuracy** | GPU available |
| **Tesseract** | ~2000ms | 85% | **CPU systems, reliability** | No GPU available |

**Auto-Selection Performance Results:**
- **Intelligent Optimization**: Automatically uses EasyOCR with GPU when available for best accuracy
- **Smart Fallback**: Falls back to Tesseract on CPU-only systems for best speed
- **Zero Configuration**: No manual setup required - works optimally out of the box
- **Hardware Awareness**: Detects and utilizes available GPU resources automatically

**What This Means for Users:**
- ✅ **Optimal Performance**: Always get the best speed/accuracy for your hardware
- ✅ **Better Accuracy**: Up to 90% vs 85% text recognition on medical forms
- ✅ **Faster Response**: GPU-accelerated processing when available
- ✅ **CPU Fallback**: Fast, reliable processing even without GPU
- ✅ **Automatic Configuration**: No technical knowledge required
- ✅ **Future-Proof**: Automatically benefits from hardware upgrades

**Technical Improvements:**
- **Smart GPU Detection**: Automatically detects CUDA-compatible GPUs using PyTorch
- **Graceful Degradation**: Falls back intelligently when hardware changes
- **Provider Caching**: Avoids expensive model reinitialization
- **Performance Logging**: Detailed logs of selection decisions and performance
- **Memory Efficient**: Optimized memory usage across all providers
- **Error Recovery**: Continues working even if preferred provider fails

**Recommendation:**
- **All Users**: Use **Auto Mode** for optimal performance without configuration
- **Advanced Users**: Can still manually select specific providers if needed
- **IT Departments**: Auto mode simplifies deployment across varied hardware configurations

### Program Shows Red/Green Boxes in Wrong Locations

This means the coordinate regions need adjustment for your specific setup. **Use the coordinate adjustment tools to fix this easily!**

**Option 1: Use the Interactive Settings Tool (Recommended)**
1. Run the settings GUI tool:
   ```
   python settings_gui.py
   ```
2. The tool will open with a screenshot of your desktop
3. First, set up the trigger region:
   - Select "trigger" from the dropdown
   - Click and drag around the "Pre-Check Rx" text or any reliable text in the window
4. Then, for each field (patient_name, prescriber_name, drug_name, direction_sig):
   - Select the field from the dropdown
   - Choose "Entered" for left panel fields or "Source" for right panel fields
   - Click and drag to draw a rectangle around the text area
   - The coordinates will be automatically updated
5. Click "Save Configuration" when done
6. Run the main program again to test

**Option 2: Manual Troubleshooting**
1. **Check Window Position**: Note the exact position and size of your pharmacy software window
2. **Use Command-Line Helper**: Run `python settings_cli.py show` to see current coordinates
3. **Test Individual Regions**: Use the settings GUI's "Test OCR" feature to verify each region
4. **Adjust as Needed**: The program adapts to any screen resolution and window configuration

---

## 📈 Advanced Usage

### Running 24/7
1. Set up coordinates using the GUI
2. Start monitoring in the web interface
3. Keep the browser tab open
4. Optional autopilot **YOLO** mode

### Multiple Configurations
- Export configurations from the GUI
- Import them on different machines
- Share setups between team members

### Integration
- Use the same `config.json` with other tools
- Parse logs programmatically
- Customize thresholds per field type

---

## 🆘 Need Help?

1. **First time?** Run `python launcher.py` and choose option 3
2. **Setup issues?** Use the GUI tool for easy coordinate setup
3. **Technical problems?** Look for error messages in the terminal
4. **Still stuck?** All tools have built-in help and validation
5. **Tired of troubleshooting?** Shoot me an Issue!

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

### 🚀 Next Phase: Advanced OCR Integration

1. I'm planning a **centralized architecture** with GPU-accelerated OCR and AI-powered semantic matching for even better accuracy while maintaining complete privacy and HIPAA compliance. The vision is a single powerful computer with a highly customized for pharmacy local AI model serving multiple pharmacies.
*My nVidia DGX reserversion has being indefinitely delayed, I'm planning to use online AI API to process the strings pulled by OCR. With prompt engineering, we can ask for a matching score of drug name and directions while keeping other PHIs locally.*  

2. By using **EasyOCR with nVIDIA CUDA**, I can leverage the GPU for much faster and more accurate OCR processing compared to CPU-bound Tesseract. EasyOCR provides excellent accuracy and supports GPU acceleration, which opens the door for faster OCR processing that can handle common fonts, layouts, and artifacts seen in prescriptions. This could significantly improve text recognition accuracy.

3. The knowledge base of current local LLM will provide the best drug name and sig **semantic matching**, replacing the hardcoded string matching logic with a more flexible and intelligent approach.

4. New opensource LLM models are coming out every month, and with the right fine-tuning and prompt engineering, I believe we can achieve very high accuracy for drug name and sig matching while keeping everything local and private.

**Central Processing Hub Vision:**
- **One Powerful Machine**: A dedicated high-performance computer with enterprise-grade GPU serving multiple pharmacy locations
- **Network-Based Service**: All pharmacies connect to the central hub for processing, eliminating the need for individual high-end workstations
- **Scalable Architecture**: One central system can handle OCR and verification requests from dozens of pharmacy workstations simultaneously
- **Cost-Effective**: Replace multiple individual CPU-limited systems with one shared powerful GPU machine
- **Centralized Management**: Single point of configuration, updates, model training and maintenance for all connected pharmacies

**Hardware Requirements:**

**Current System (CPU-Based per Pharmacy Station):**
- **Minimum**: 8GB RAM, modern CPU for basic OCR processing per workstation
- **Limitation**: CPU-intensive Tesseract OCR, hardcoded text matching, requires individual setup at each location, rate limited by the computer's CPU power

**Future System (Centralized GPU Hub):**
- **Central Hub Minimum**: 32GB RAM, RTX 5070+ for serving 5-10 pharmacy workstations
- **Central Hub Recommended**: 64GB RAM, RTX 5090 for serving 15-25 pharmacy workstations  
- **Central Hub Optimal**: 128GB+ RAM, DGX/H100 for serving 50+ pharmacy workstations
- **Pharmacy Workstations**: Basic computers with network connectivity - no special hardware required
- **Network Infrastructure**: Reliable high-speed internet connection between pharmacies and central hub

**Implementation Timeline:**
- **Phase 1 (Centralized EasyOCR)**: convert to EasyOCR for GPU-based OCR processing, build initial central hub prototype
- **Phase 1.5 (Local LLM Integration)**: build local LLM for semantic drug name and sig matching, replacing hardcoded logic
- **Phase 2 (Network Architecture)**: Develop secure API system for pharmacy workstations to communicate with central hub
- **Phase 3 (AI Semantic Analysis)**: Replace hardcoded string matching with LLM-based semantic understanding on the central system with prompt engineering
- **Phase 3.5 (Local LLM Training)**: Fine-tune local LLM on pharmacy-specific data for better accuracy
- **Phase 4 (Multi-Pharmacy Integration)**: Scale to support dozens of pharmacy locations from single central hub
- **Phase 5 (Enterprise Features)**: Add centralized reporting, analytics, and management across all connected pharmacies
- This represents an evolution from individual pharmacy systems to a centralized service architecture
- Community feedback and pilot testing with pharmacy chains will guide development priorities
- Maintain backward compatibility with current individual pharmacy installations

**Why This Centralized Approach:**
- **Cost Efficiency**: One powerful machine replaces dozens of individual high-end workstations
- **Performance**: Enterprise-grade GPU processing delivers consistent high-speed OCR for all connected pharmacies
- **Scalability**: Easy to add new pharmacy locations without additional hardware investments
- **Maintenance**: Single point of updates, configuration, and troubleshooting
- **Reliability**: Professional-grade central system with redundancy and backup capabilities
- **Analytics**: Aggregated insights across multiple pharmacy locations while maintaining privacy

*Interested in this direction? Feel free to reach out to me,submit issue or just send a pull request. Let's make pharmacy work fully AI-lized*

---
