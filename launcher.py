#!/usr/bin/env python3
"""
Pharmacy Verification System Launcher
=====================================

Complete launcher for the Pharmacy Verification System with automatic setup,
OCR engine configuration, and easy access to all components.

Usage:
    python launcher.py
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit',
        'tkinter',  # Usually built-in with Python
        'pyautogui',
        'pytesseract',
        'PIL',
        'rapidfuzz'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'tkinter':
                import tkinter
            else:
                __import__(package)
        except ImportError:
            if package == 'PIL':
                missing_packages.append('pillow')
            else:
                missing_packages.append(package)
    
    return missing_packages

def check_ocr_engines():
    """Check which OCR engines are available"""
    engines = {
        'easyocr': False, 
        'tesseract': False
    }
    
    # Check EasyOCR
    try:
        import easyocr
        engines['easyocr'] = True
    except ImportError:
        pass
    
    # Check Tesseract
    try:
        import pytesseract
        engines['tesseract'] = True
    except ImportError:
        pass
    
    return engines

def install_packages(packages):
    """Install missing packages with smart pip detection"""
    if not packages:
        return True
        
    print(f"📦 Installing missing packages: {', '.join(packages)}")
    
    # Try different pip commands
    pip_commands = [
        [sys.executable, '-m', 'pip', 'install'],
        ['python3', '-m', 'pip', 'install'],
        ['pip3', 'install'],
        ['pip', 'install']
    ]
    
    for pip_cmd in pip_commands:
        try:
            subprocess.check_call(pip_cmd + packages, 
                                stdout=subprocess.DEVNULL, 
                                stderr=subprocess.DEVNULL)
            print("✅ Installation successful!")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print("❌ Installation failed. Please install manually:")
    print(f"   pip install {' '.join(packages)}")
    return False

def install_easyocr():
    """Install EasyOCR with error handling"""
    print("\n🚀 Installing EasyOCR (recommended OCR engine)...")
    print("   ⏳ This may take a few minutes...")
    
    pip_commands = [
        [sys.executable, '-m', 'pip', 'install'],
        ['python3', '-m', 'pip', 'install'],
        ['pip3', 'install'],
        ['pip', 'install']
    ]
    
    for i, pip_cmd in enumerate(pip_commands, 1):
        print(f"   🔄 Trying installation method {i}/{len(pip_commands)}...")
        try:
            subprocess.check_call(pip_cmd + ['easyocr'])
            print("✅ EasyOCR installed successfully!")
            
            # Test the installation
            try:
                import easyocr
                reader = easyocr.Reader(['en'], gpu=False)  # CPU mode for compatibility
                print("✅ EasyOCR test successful!")
                return True
            except Exception as e:
                print(f"⚠️  EasyOCR installation test failed: {e}")
                return False
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            if i < len(pip_commands):
                print(f"   ❌ Method {i} failed, trying next...")
                continue
            else:
                print("❌ All installation methods failed.")
                break
    
    print("\n💡 Manual installation instructions:")
    print("   pip install easyocr")
    return False

def update_config_for_ocr(ocr_engine):
    """Update config.json to use specified OCR engine"""
    config_path = Path("config/config.json")
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except json.JSONDecodeError:
            print("⚠️  Invalid config.json, creating new configuration...")
            config = {}
    else:
        print("📝 Creating new configuration file...")
        config = {}
    
    # Update OCR provider
    config["ocr_provider"] = ocr_engine
    
    # Add engine-specific settings
    if ocr_engine == "easyocr":
        config.setdefault("easyocr", {})
        config["easyocr"]["use_gpu"] = True
        config["easyocr"]["confidence_threshold"] = 0.5
    elif ocr_engine == "auto":
        # Set up defaults for both engines when using auto mode
        config.setdefault("easyocr", {})
        config["easyocr"]["use_gpu"] = True  # Will be auto-detected
        config["easyocr"]["confidence_threshold"] = 0.5
        config.setdefault("tesseract", {})
        config["tesseract"]["config_options"] = "--psm 7"
        config["tesseract"]["fallback_config"] = "--psm 8"
    
    # Save configuration
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        if ocr_engine == "auto":
            print(f"✅ Configuration updated to use AUTO selection (smart GPU detection)")
        else:
            print(f"✅ Configuration updated to use {ocr_engine.upper()}")
        return True
    except Exception as e:
        print(f"❌ Failed to update configuration: {e}")
        return False

def setup_ocr_engine(auto_select=False):
    """Interactive OCR engine setup"""
    print("\n🔍 Checking OCR engines...")
    engines = check_ocr_engines()
    
    available_engines = [name for name, available in engines.items() if available]
    
    if not available_engines:
        print("❌ No OCR engines found!")
        print("\n📦 Would you like to install EasyOCR (recommended)?")
        print("   • Good accuracy on medical forms")
        print("   • GPU acceleration support")
        
        choice = input("\nInstall EasyOCR? (y/n): ").lower().strip()
        if choice == 'y':
            if install_easyocr():
                update_config_for_ocr("easyocr")
                return True
        
        print("\n💡 Alternative: Install Tesseract manually")
        print("   pip install pytesseract")
        return False
    
    print("\n📋 Available OCR Options:")
    available_options = []
    option_num = 1
    
    # Always show auto option first
    print(f"   {option_num}. AUTO: ✅ Smart selection (GPU detection, best performance)")
    available_options.append("auto")
    option_num += 1
    
    for engine in ['easyocr', 'tesseract']:
        if engines[engine]:
            status = "✅ Installed"
            performance = {"easyocr": "Good accuracy, easy install",
                          "tesseract": "Very reliable, CPU-optimized"}[engine]
            print(f"   {option_num}. {engine.upper()}: {status} ({performance})")
            available_options.append(engine)
            option_num += 1
        else:
            status = "❌ Not installed"
            performance = {"easyocr": "Good accuracy, easy install",
                          "tesseract": "Very reliable, CPU-optimized"}[engine]
            print(f"   -. {engine.upper()}: {status} ({performance})")
    
    # If auto_select is True (first-time setup), use auto mode
    if auto_select:
        print(f"\n🎯 Auto-selecting AUTO mode (smart OCR selection)")
        update_config_for_ocr("auto")
        return True
    
    # Manual selection for explicit OCR setup
    print(f"\n🎯 Select OCR Engine (1-{len(available_options)}):")
    
    try:
        choice = input("Enter your choice: ").strip()
        choice_num = int(choice)
        
        if 1 <= choice_num <= len(available_options):
            selected_engine = available_options[choice_num - 1]
            # Convert engine name to config format
            config_name = {'auto': 'auto', 'easyocr': 'easyocr', 'tesseract': 'tesseract'}[selected_engine]
            
            print(f"\n✅ Selected {selected_engine.upper()}")
            update_config_for_ocr(config_name)
            return True
        else:
            print(f"❌ Invalid choice. Please select 1-{len(available_options)}")
            return False
            
    except ValueError:
        print("❌ Invalid input. Please enter a number.")
        return False

def main():
    print("🚀 Pharmacy Verification System Launcher")
    print("=" * 50)
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} is too old. Need Python 3.7+")
        print("   Please upgrade Python and try again.")
        return
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    
    # Check basic dependencies
    print("\n📦 Checking basic dependencies...")
    missing = check_dependencies()
    
    if missing:
        print(f"⚠️  Missing packages: {', '.join(missing)}")
        install_choice = input("📥 Install missing packages? (y/n): ").lower().strip()
        
        if install_choice == 'y':
            if not install_packages(missing):
                return
        else:
            print("❌ Cannot continue without required packages.")
            return
    else:
        print("✅ Basic dependencies are installed!")
    
    # Setup OCR engine
    print("\n🔧 Setting up OCR engine...")
    if not setup_ocr_engine(auto_select=True):
        print("❌ OCR setup failed. Some features may not work properly.")
    
    # Check system status
    config_exists = os.path.exists('config/config.json')
    gui_exists = os.path.exists('ui/settings_gui.py')
    
    print("\n📋 System Status:")
    print(f"   📄 Config file: {'✅ Found' if config_exists else '❌ Missing'}")
    print(f"   🛠️  Setup GUI: {'✅ Available' if gui_exists else '❌ Missing'}")
    
    # Show current OCR engine
    if config_exists:
        try:
            with open('config/config.json', 'r') as f:
                config = json.load(f)
                ocr_provider = config.get('ocr_provider', 'unknown')
                print(f"   🔍 OCR Engine: ✅ {ocr_provider.upper()}")
        except:
            print(f"   🔍 OCR Engine: ⚠️  Configuration error")
    
    if not config_exists:
        print("\n⚠️  No configuration found!")
        print("   You can set up coordinates using the web interface.")
    
    print("\n🚀 Launching Streamlit Web Interface...")
    print("📍 Your browser should open automatically to: http://localhost:8501")
    print("🛠️  All settings and configuration are available in the web interface")
    print("💡 Tip: Use the 'Settings' tab in the web interface to configure coordinates and OCR")
    
    # Launch Streamlit directly
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'ui/streamlit_app.py'])

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 Try running the components manually:")
        print("   • For monitoring: streamlit run ui/streamlit_app.py")
        print("   • For setup: python ui/settings_gui.py")
