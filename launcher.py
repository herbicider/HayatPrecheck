#!/usr/bin/env python3
"""
Quick launcher for the Simple Pharmacy Verification System
=========================================================

This script provides easy access to both the Streamlit monitoring interface
and the proven coordinate setup GUI.

Usage:
    python launch_simple.py
"""

import subprocess
import sys
import os
import time

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

def install_packages(packages):
    """Install missing packages"""
    if not packages:
        return True
        
    print(f"Installing missing packages: {', '.join(packages)}")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
        return True
    except subprocess.CalledProcessError:
        print("Failed to install packages. Please install manually:")
        print(f"pip install {' '.join(packages)}")
        return False

def main():
    print("🚀 Simple Pharmacy Verification System Launcher")
    print("=" * 50)
    
    # Check dependencies
    print("📦 Checking dependencies...")
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
        print("✅ All dependencies are installed!")
    
    # Check if config exists
    config_exists = os.path.exists('config.json')
    gui_exists = os.path.exists('settings_gui.py')
    
    print("\n📋 System Status:")
    print(f"   📄 Config file: {'✅ Found' if config_exists else '❌ Missing'}")
    print(f"   🛠️  Setup GUI: {'✅ Available' if gui_exists else '❌ Missing'}")
    
    if not config_exists:
        print("\n⚠️  No configuration found!")
        print("   You'll need to set up coordinates using the GUI first.")
    
    print("\n🎯 What would you like to do?")
    print("   1. 📊 Launch Streamlit Monitor (web interface)")
    print("   2. 🛠️  Launch Coordinate Setup GUI")
    print("   3. 🚀 Launch Both (recommended for first-time setup)")
    print("   4. ❌ Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        print("\n🌐 Starting Streamlit web interface...")
        print("📍 Your browser should open automatically to: http://localhost:8501")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])
        
    elif choice == '2':
        if not gui_exists:
            print("❌ Setup GUI not found (settings_gui.py missing)")
            return
        print("\n🛠️  Starting coordinate setup GUI...")
        subprocess.run([sys.executable, 'settings_gui.py'])
        
    elif choice == '3':
        if not gui_exists:
            print("❌ Setup GUI not found (settings_gui.py missing)")
            return
            
        print("\n🚀 Starting both interfaces...")
        print("   🛠️  First: Coordinate Setup GUI")
        print("   🌐 Then: Streamlit Monitor")
        
        # Start GUI first
        gui_process = subprocess.Popen([sys.executable, 'settings_gui.py'])
        
        # Give GUI time to start
        time.sleep(2)
        
        # Start Streamlit
        print("📍 Opening Streamlit in your browser...")
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py'])
        
    elif choice == '4':
        print("👋 Goodbye!")
        
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("💡 Try running the components manually:")
        print("   • For monitoring: streamlit run streamlit_app.py")
        print("   • For setup: python settings_gui.py")
