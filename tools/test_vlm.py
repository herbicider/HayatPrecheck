#!/usr/bin/env python3
"""
VLM Integration Test Script
==========================

Simple test to verify VLM functionality is working correctly.

Usage:
    python test_vlm.py
"""

import json
import os
import sys
import time
from pathlib import Path

def test_vlm_config():
    """Test VLM configuration file"""
    print("📋 Testing VLM Configuration...")
    
    vlm_config_file = os.path.join("config", "vlm_config.json")
    
    if not os.path.exists(vlm_config_file):
        print(f"❌ {vlm_config_file} not found")
        return False
    
    try:
        with open(vlm_config_file, 'r') as f:
            config = json.load(f)
        
        required_sections = ["vlm_config", "vlm_regions", "vlm_settings"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing section: {section}")
                return False
        
        print("✅ VLM configuration file is valid")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {vlm_config_file}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading {vlm_config_file}: {e}")
        return False

def test_vlm_import():
    """Test VLM verifier import"""
    print("📦 Testing VLM Import...")
    
    try:
        from ai.vlm_verifier import VLM_Verifier
        print("✅ VLM Verifier imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import VLM Verifier: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing VLM Verifier: {e}")
        return False

def test_dependencies():
    """Test required dependencies"""
    print("🔍 Testing Dependencies...")
    
    required = {
        'openai': 'OpenAI client',
        'PIL': 'Pillow image processing',
        'pyautogui': 'Screen capture',
        'streamlit': 'Web interface'
    }
    
    missing = []
    
    for package, description in required.items():
        try:
            if package == 'PIL':
                import PIL
            else:
                __import__(package)
            print(f"✅ {description}")
        except ImportError:
            print(f"❌ {description} (missing: {package})")
            missing.append(package)
    
    if missing:
        print(f"\n💡 Install missing packages:")
        if 'PIL' in missing:
            missing[missing.index('PIL')] = 'pillow'
        print(f"   pip install {' '.join(missing)}")
        return False
    
    return True

def test_vlm_connection():
    """Test VLM model connection"""
    print("🔗 Testing VLM Connection...")
    
    try:
        from ai.vlm_verifier import VLM_Verifier
        
        # Load config
        with open(os.path.join("config", "vlm_config.json"), 'r') as f:
            vlm_config = json.load(f)
        
        # Create verifier
        verifier = VLM_Verifier(vlm_config)
        
        # Test connection
        result = verifier.test_vlm_connection()
        
        if result.get("success"):
            print("✅ VLM connection successful!")
            print(f"   Model: {result.get('model')}")
            print(f"   Response: {result.get('response', 'No response')[:50]}...")
            return True
        else:
            print("❌ VLM connection failed!")
            print(f"   Error: {result.get('error')}")
            print(f"   Model: {result.get('model')}")
            print(f"   URL: {result.get('base_url')}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing VLM connection: {e}")
        return False

def test_main_config():
    """Test main configuration has VLM mode option"""
    print("⚙️ Testing Main Configuration...")
    
    config_file = "config.json"
    
    if not os.path.exists(config_file):
        print(f"⚠️  {config_file} not found - will be created by application")
        return True
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        verification_mode = config.get("verification_mode", "ocr")
        print(f"✅ Current verification mode: {verification_mode}")
        
        if verification_mode == "vlm":
            print("✅ VLM mode is active")
        else:
            print("✅ OCR mode is active (can switch to VLM in Streamlit)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading {config_file}: {e}")
        return False

def run_integration_test():
    """Run a quick integration test"""
    print("🧪 Running Integration Test...")
    
    try:
        # Test VLM screenshot capture
        from ai.vlm_verifier import VLM_Verifier
        
        with open(os.path.join("config", "vlm_config.json"), 'r') as f:
            vlm_config = json.load(f)
        
        verifier = VLM_Verifier(vlm_config)
        
        # Test region capture
        print("   📸 Testing screenshot capture...")
        
        data_entry_img = verifier.capture_region_screenshot("data_entry")
        source_img = verifier.capture_region_screenshot("source")
        
        if data_entry_img and source_img:
            print("   ✅ Screenshot capture successful")
            print(f"   📊 Data entry image: {data_entry_img.size}")
            print(f"   📊 Source image: {source_img.size}")
            return True
        else:
            print("   ❌ Screenshot capture failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    print("🔬 VLM Integration Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("VLM Config", test_vlm_config),
        ("VLM Import", test_vlm_import),
        ("Main Config", test_main_config),
    ]
    
    # Run basic tests
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
    
    print(f"\n📊 Basic Tests: {passed}/{total} passed")
    
    # Run connection test if basic tests pass
    if passed == total:
        print("\n🔗 Advanced Tests:")
        
        connection_passed = test_vlm_connection()
        if connection_passed:
            integration_passed = run_integration_test()
            
            if integration_passed:
                print("\n🎉 All Tests Passed!")
                print("✅ VLM integration is ready to use")
                print("\n💡 Next steps:")
                print("   1. Launch Streamlit: streamlit run streamlit_app.py")
                print("   2. Go to VLM Configuration page")
                print("   3. Set up screenshot regions")
                print("   4. Switch to VLM mode in Monitor page")
            else:
                print("\n⚠️  Integration test failed")
                print("💡 Check VLM region coordinates in VLM Configuration")
        else:
            print("\n⚠️  Connection test failed")
            print("💡 Check if your VLM server is running:")
            print("   • Ollama: ollama serve")
            print("   • Other: verify base_url in config/vlm_config.json")
    else:
        print("\n❌ Basic tests failed")
        print("💡 Fix the failed tests before proceeding")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
