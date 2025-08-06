# Pharmacy Pre-Check Verification Agent

The **simplest** way to set up and monitor your pharmacy verification system! This program watches your screen for the "Pre-Check Rx" pharmacy software and helps verify that the information entered matches the source document. It places colored boxes over fields to show you if they match (green) or don't match (red).

**🚀 EASY SETUP:** Use the simple launcher to get started with both the coordinate setup GUI and web monitoring interface!

## 🎯 Quick Start (Recommended)

### Option 1: Super Easy Launcher (Recommended)
```bash
python launcher.py
```
**Or double-click:** `start.bat` (Windows)

Choose option 3 to launch both the setup GUI and monitoring interface!

### Option 2: Manual Steps
1. **Set up coordinates:** `python settings_gui.py`
2. **Monitor system:** `streamlit run streamlit_app.py`

## 📖 What You Get

- **🛠️ Easy Setup GUI**: Drag and drop to select screen regions
- **🌐 Web Monitoring**: Modern dashboard accessible from any device
- **📱 Mobile Access**: Check status from your phone or tablet
- **📊 Real-time Analytics**: Performance metrics and accuracy tracking
- **🔍 Live OCR Testing**: Verify your setup immediately

---

## 🎯 Why This Approach?

| Feature | This System | Complex Solutions |
|---------|-------------|-------------------|
| **Setup** | ✅ Drag & drop GUI | ❌ Manual coordinates |
| **Monitoring** | ✅ Web dashboard | ❌ Command line only |
| **Mobile Access** | ✅ Yes | ❌ No |
| **User Friendly** | ✅ Very easy | ❌ Technical |
| **Reliability** | ✅ Proven GUI + modern web | ❌ Experimental |

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

### 🚀 Super Quick Setup

1. **Download and extract** this project
2. **Run the launcher:**
   ```bash
   python launcher.py
   ```
3. **Choose option 3** (Launch Both)
4. **Set up coordinates** in the GUI that opens
5. **Monitor** in the web browser that opens

### Manual Setup Steps

Follow these steps if you prefer to do things manually.

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

### Step 4: Set Up the OCR Engine (Tesseract)

This is the engine that reads the text on your screen.

1.  Download the **Tesseract OCR** for Windows from this link: [Download Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/releases)
    *   Find the most recent release and download the Windows installer (`.exe` file) that matches your system (32-bit or 64-bit).
2.  Run the downloaded installer and follow the installation wizard.
    *   **Important:** During installation, make sure to check the option to "Add Tesseract to PATH" if available, or note the installation directory (usually `C:\Program Files\Tesseract-OCR`).
3.  After installation is complete, Tesseract will be available system-wide.

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

All settings are managed through the GUI - no manual file editing required!

The system automatically:
- ✅ Creates configuration files
- ✅ Validates coordinate setup  
- ✅ Backs up settings
- ✅ Tests OCR functionality

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

## 🌐 Streamlit Web Interface

**🆕 NEW!** We now offer a modern web-based interface that combines monitoring and configuration in one easy-to-use application.

### Quick Start with Streamlit

#### Option 1: Auto-Launcher (Recommended)
- **Any OS**: Run `python launcher.py` and choose option 3

#### Option 2: Manual Launch
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Features

🏠 **Single Interface**: Everything in one web application
📊 **Real-time Monitoring**: Live charts and analytics of verification performance  
⚙️ **Visual Settings**: Easy configuration with sliders and visual feedback
📱 **Mobile Friendly**: Access from any device on your network
📈 **Advanced Analytics**: Time-based filtering and performance metrics
🔧 **Coordinate Setup**: Screenshot-assisted coordinate configuration
💾 **Auto-Backup**: Automatic configuration backups
🎯 **OCR Testing**: Test regions directly in the interface

### Access Your Application
Once launched, open your web browser to: http://localhost:8501

---

## 🆘 Troubleshooting

### "tesseract is not installed or it's not in your PATH" Error

If you see this error, it means the program can't find your Tesseract installation. Try these solutions:

**Option 1: Add Tesseract to your system PATH**
1. Find where Tesseract was installed (usually `C:\Program Files\Tesseract-OCR`)
2. Add this path to your Windows PATH environment variable:
   - Press `Windows + R`, type `sysdm.cpl`, and press Enter
   - Click "Environment Variables"
   - Under "System Variables", find and select "Path", then click "Edit"
   - Click "New" and add the Tesseract installation path (e.g., `C:\Program Files\Tesseract-OCR`)
   - Click "OK" on all windows
3. Restart your Command Prompt and try running the program again

### "GUI won't start"
- Make sure you have Python with tkinter: `python -c "import tkinter"`
- On some Linux systems: `sudo apt-get install python3-tk`

### "Web interface won't open"
- Try: `python -m streamlit run streamlit_app.py`
- Check firewall isn't blocking port 8501
- Try `http://127.0.0.1:8501` instead

### "Can't capture coordinates"
- **Windows:** Run as administrator if needed
- **macOS:** Grant screen recording permission in System Preferences
- **Linux:** Install `python3-tk` and `scrot`

### "OCR not working"
- Install Tesseract: Download from https://github.com/tesseract-ocr/tesseract
- Windows: Add to PATH or put in project folder
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

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

## 📁 File Structure

After cleanup, here's what each file does:

| File | Purpose |
|------|---------|
| `launcher.py` | **Start here!** Main launcher with setup wizard |
| `start.bat` | **Windows users:** Double-click to launch |
| `settings_gui.py` | **Coordinate setup:** Drag & drop region selection |
| `streamlit_app.py` | **Web monitoring:** Modern dashboard interface |
| `Precheck_OCR.py` | **Core engine:** The verification logic |
| `settings_manager.py` | **Configuration:** Manages settings and backups |
| `config.json` | **Your settings:** Coordinates and thresholds |

---

## 📈 Advanced Usage

### Running 24/7
1. Set up coordinates using the GUI
2. Start monitoring in the web interface
3. Keep the browser tab open
4. Optionally run as a system service

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
5. **Monitor logs** to fine-tune accuracy

---

## 📈 Future Improvements

We're planning GPU-accelerated OCR and AI-powered semantic matching for even better accuracy while maintaining complete privacy and HIPAA compliance. All processing will remain local to your machine.
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

**💡 Pro Tip:** Start with the launcher script (`python launcher.py`) and choose option 3 to get both interfaces running perfectly!

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

All settings are managed through the GUI - no manual file editing required!

The system automatically:
- ✅ Creates configuration files
- ✅ Validates coordinate setup  
- ✅ Backs up settings
- ✅ Tests OCR functionality

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

## 🌐 Streamlit Web Interface

**🆕 NEW!** We now offer a modern web-based interface that combines monitoring and configuration in one easy-to-use application.

### Quick Start with Streamlit

#### Option 1: Auto-Launcher (Recommended)
- **Any OS**: Run `python launcher.py` and choose option 3

#### Option 2: Manual Launch
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Streamlit Features

🏠 **Single Interface**: Everything in one web application
📊 **Real-time Monitoring**: Live charts and analytics of verification performance  
⚙️ **Visual Settings**: Easy configuration with sliders and visual feedback
📱 **Mobile Friendly**: Access from any device on your network
📈 **Advanced Analytics**: Time-based filtering and performance metrics
🔧 **Coordinate Setup**: Screenshot-assisted coordinate configuration
💾 **Auto-Backup**: Automatic configuration backups
🎯 **OCR Testing**: Test regions directly in the interface

### Access Your Application
Once launched, open your web browser to: http://localhost:8501

---

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

## 🌐 Web Interface Features

The modern web interface (launched via `streamlit run streamlit_app.py`) provides:

- **📊 Real-time Monitoring**: Live dashboard showing verification status
- **� Performance Analytics**: Charts and metrics showing accuracy over time  
- **� Mobile Access**: Monitor from any device on your network
- **🔄 Easy Controls**: Start/stop monitoring with one click
- **📋 Live Logs**: See verification results as they happen
- **🎯 Field Analysis**: Identify which fields need attention

Access the web interface at: `http://localhost:8501`

For detailed web interface documentation, see [STREAMLIT_README.md](STREAMLIT_README.md)

---

## 📁 File Structure

After cleanup, here's what each file does:

| File | Purpose |
|------|---------|
| `launcher.py` | **Start here!** Main launcher with setup wizard |
| `start.bat` | **Windows users:** Double-click to launch |
| `settings_gui.py` | **Coordinate setup:** Drag & drop region selection |
| `streamlit_app.py` | **Web monitoring:** Modern dashboard interface |
| `Precheck_OCR.py` | **Core engine:** The verification logic |
| `settings_manager.py` | **Configuration:** Manages settings and backups |
| `config.json` | **Your settings:** Coordinates and thresholds |
| `STREAMLIT_README.md` | **Web interface guide:** Detailed documentation |

---

## 🆘 Need Help?

1. **First time?** Run `python launcher.py` and choose option 3
2. **Setup issues?** Check the detailed documentation in `STREAMLIT_README.md`
3. **Technical problems?** Look for error messages in the terminal
4. **Still stuck?** All tools have built-in help and validation

---

## 🔒 Privacy & Security

- ✅ **Completely local** - no data sent anywhere
- ✅ **HIPAA friendly** - patient data stays on your machine  
- ✅ **No internet required** for core functionality
- ✅ **Open source** - you can see exactly what it does

---

## 📈 Future Improvements

We're planning GPU-accelerated OCR and AI-powered semantic matching for even better accuracy while maintaining complete privacy and HIPAA compliance. All processing will remain local to your machine.
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
