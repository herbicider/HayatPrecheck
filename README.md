# Pharmacy Pre-Check Verification Agent

This program watches your screen for the "Pre-Check Rx" pharmacy software and helps verify that the information entered matches the source document. It places colored boxes over fields to show you if they match (green) or don't match (red).

**SETUP REQUIRED:** The program includes coordinate adjustment tools that let you configure it for any screen resolution or setup. Simply use the coordinate adjuster tool to define the regions for your specific pharmacy software layout.

## How It Works

The program automatically handles common differences between the entered data and source documents:

- **Name Format Normalization**: The left side shows names as "Last, First" while the right side shows "First Last" - the program automatically converts both to the same format for accurate comparison
- **Pharmacy Abbreviation Expansion**: Common pharmacy abbreviations (po, bid, tid, etc.) are expanded for better matching
- **Smart Text Cleaning**: Removes punctuation and normalizes spacing for consistent comparisons
- **Flexible Matching**: Uses a 65% similarity threshold to account for minor OCR differences and typos
- **Adaptive Timing**: Responds quickly to screen changes (0.1s) and slows down when screen is static to save CPU
- **Prescription Change Detection**: Automatically detects when you move to a new prescription and re-runs verification

---

## Setup Instructions

Follow these steps exactly to get the program running on your computer.

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

1.  Open the **Command Prompt**. You can find it by clicking the Start Menu and typing `cmd`.
2.  Navigate to the project folder you extracted in Step 1. Type `cd` followed by the path to the folder. For example, if it's on your desktop, you would type:
    ```
    cd C:\Users\YourUsername\Desktop\HayatPrecheck-main
    ```
3.  Once you are in the correct folder, copy and paste the following command into the Command Prompt and press **Enter**:
    ```
    pip install -r requirements.txt
    ```
    This will automatically install all the necessary libraries for the program.

### Step 4: Set Up the OCR Engine (Tesseract)

This is the engine that reads the text on your screen.

1.  Download the **Tesseract OCR** for Windows from this link: [Download Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/releases)
    *   Find the most recent release and download the Windows installer (`.exe` file) that matches your system (32-bit or 64-bit).
2.  Run the downloaded installer and follow the installation wizard.
    *   **Important:** During installation, make sure to check the option to "Add Tesseract to PATH" if available, or note the installation directory (usually `C:\Program Files\Tesseract-OCR\`).
3.  After installation is complete, Tesseract will be available system-wide.

Your project folder should now look like this:
```
HayatPrecheck-main/
|-- Precheck_OCR.py              # Main verification program
|-- settings_gui.py              # Enhanced GUI settings and coordinate tool
|-- settings_cli.py              # Command-line settings and coordinate helper
|-- launcher.sh                  # Unix/macOS launcher script
|-- launcher.bat                 # Windows launcher script
|-- config.json                  # Configuration file with coordinates
|-- requirements.txt             # Python dependencies
|-- verification.log             # Program log file (created when running)
|-- README.md                    # This file
```

---

## How to Run the Program

**SETUP NOTES:**
- The program works with any screen resolution and window configuration
- Use the coordinate adjustment tools (described below) to configure regions for your setup
- Your pharmacy software can be windowed or maximized - just adjust coordinates accordingly

1.  Open your pharmacy software in your preferred window configuration
2.  Open the Command Prompt and navigate to the project folder (as you did in Step 3.2)
3.  Type the following command and press **Enter**:
    ```
    python Precheck_OCR.py
    ```
4.  The agent is now running. It will silently watch your screen. When it sees the "Pre-Check Rx" window, it will automatically perform the check and show the colored boxes
5.  **If the colored boxes appear in wrong locations:** Stop the program (Ctrl+C) and use the coordinate adjustment tools below to configure the regions for your setup

## How to Stop the Program

To stop the program, go to the Command Prompt window where it is running and press the **`Ctrl`** + **`C`** keys at the same time.

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
- If GUI doesn't work, install with: `brew install python-tk` (macOS)

### Command-Line Tool (Fallback)

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

- **Patient Name Threshold** (default: 65%): How similar patient names must be to consider a match
- **Prescriber Name Threshold** (default: 65%): How similar prescriber names must be to consider a match  
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

## Troubleshooting

### "tesseract is not installed or it's not in your PATH" Error

If you see this error, it means the program can't find your Tesseract installation. Try these solutions:

**Option 1: Add Tesseract to your system PATH**
1. Find where Tesseract was installed (usually `C:\Program Files\Tesseract-OCR\`)
2. Add this path to your Windows PATH environment variable:
   - Press `Windows + R`, type `sysdm.cpl`, and press Enter
   - Click "Environment Variables"
   - Under "System Variables", find and select "Path", then click "Edit"
   - Click "New" and add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR\`)
   - Click "OK" on all windows
3. Restart your Command Prompt and try running the program again

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

**Option 2: Create a local tesseract folder (if Option 1 doesn't work)**
1. In your project folder, create a new folder named `tesseract`
2. Copy `tesseract.exe` and the `tessdata` folder from your Tesseract installation directory to this new folder
3. Your project should then look like this:
   ```
   HayatPrecheck-main/
   |-- tesseract/
   |   |-- tesseract.exe
   |   |-- tessdata/
   |-- Precheck_OCR.py
   |-- settings_gui.py
   |-- settings_cli.py
   |-- launcher.sh
   |-- launcher.bat
   |-- config.json
   |-- requirements.txt
   |-- README.md
   ```

---

## Future Roadmap

### 🤖 Local LLM Integration (Planned)

**Vision:** Replace traditional CPU-heavy OCR and hardcoded text comparison with GPU-accelerated vision processing and local Large Language Model semantic analysis for enhanced accuracy and HIPAA compliance.

**Benefits:**
- **🔒 HIPAA Compliant**: All patient data stays on your local machine - no cloud services
- **🚀 GPU Acceleration**: Use PaddleOCR with dedicated GPU hardware for faster, more accurate text recognition
- **🧠 Semantic Understanding**: LLMs understand context, medical terminology, and intent rather than just text similarity
- **📋 Intelligent Comparison**: Understands equivalent expressions, abbreviations, and medical context
- **🎯 Reduced False Positives**: Recognizes when "twice daily" = "BID" = "2x/day" without hardcoded rules
- **📝 Natural Language Processing**: Handles complex medication instructions and dosages intelligently
- **🔄 Self-Improving**: Can learn from corrections and adapt to specific pharmacy workflows
- **💪 Purpose-Built Hardware**: Designed for dedicated high-performance PCs with strong GPUs

**Planned Features:**
- **PaddleOCR Integration**: GPU-accelerated OCR engine replacing CPU-intensive Tesseract
- **Local LLM Models**: Integration with Ollama, LLaMA, or similar local models for semantic analysis
- **Medical Domain Knowledge**: Specialized models trained on pharmaceutical terminology and workflows
- **Vision-Language Models**: Direct image understanding with contextual text analysis
- **Semantic Comparison**: Replace hardcoded string matching with intelligent meaning-based validation
- **Multi-GPU Support**: Utilize powerful dedicated hardware for real-time processing
- **Privacy-First Design**: Zero external API calls, complete data isolation

**Technical Approach:**
- **Phase 1**: Replace Tesseract with PaddleOCR for GPU-accelerated text recognition
- **Phase 2**: Integrate local vision-language models (LLaVA, Moondream) for image understanding
- **Phase 3**: Replace hardcoded matching algorithms with LLM semantic comparison
- **Phase 4**: Add specialized medical language models for pharmacy-specific validation
- **Phase 5**: Implement learning mechanisms and advanced features like dosage calculation validation

**Hardware Strategy:**
- **Current Setup**: Works on standard business computers with weak GPUs using CPU-based processing
- **Future Setup**: Dedicated high-performance PC with strong GPU specifically for pharmacy automation
- **Deployment**: Central powerful machine can serve multiple pharmacy workstations via network
- **Cost-Effective**: One powerful GPU machine can replace multiple CPU-limited workstations

**Hardware Requirements:**

**Current System (CPU-Based):**
- **Minimum**: 8GB RAM, modern CPU for basic OCR processing
- **Limitation**: CPU-intensive Tesseract OCR, hardcoded text matching

**Future System (GPU-Accelerated):**
- **Minimum**: 16GB RAM, RTX 4060 or better for PaddleOCR + basic LLM
- **Recommended**: 32GB RAM, RTX 4070+ for real-time vision-language processing  
- **Optimal**: 64GB+ RAM, RTX 4090 or professional GPU for multiple simultaneous workstations
- **Purpose-Built**: Dedicated automation PC with high-end GPU serving entire pharmacy

**Implementation Timeline:**
- **Phase 1 (PaddleOCR)**: Replace Tesseract with GPU-accelerated OCR for immediate performance gains
- **Phase 2 (Semantic Analysis)**: Replace hardcoded string matching with LLM-based semantic understanding
- **Phase 3 (Integration)**: Combine vision and language models for comprehensive pharmacy automation
- This represents a significant evolution from CPU-limited to GPU-powered intelligent systems
- Community feedback and pilot testing in pharmacy environments will guide development priorities
- Maintain backward compatibility with current CPU-based system for existing installations

**Why This Approach:**
- **Performance**: GPU processing is 10-100x faster than CPU for vision tasks
- **Accuracy**: Semantic understanding vs. rigid text matching reduces false positives/negatives
- **Scalability**: One powerful machine can serve multiple pharmacy workstations
- **Future-Proof**: Built for the next generation of AI-powered pharmacy automation

*Interested in this direction? Let us know your thoughts and specific use cases!*

---
