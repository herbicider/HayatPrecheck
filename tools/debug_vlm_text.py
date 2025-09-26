#!/usr/bin/env python3
"""
VLM Text Extraction Debug Tool
==============================

This tool helps debug what text the VLM can actually see in prescription images.
Useful for troubleshooting VLM verification accuracy issues.

Usage:
    python tools/debug_vlm_text.py

The tool will:
1. Capture screenshots of both VLM regions (data_entry and source)
2. Ask the VLM to extract all text it can see from each image
3. Display the results for comparison with expected text

This helps identify if VLM verification issues are due to:
- Poor image quality/enhancement
- VLM model limitations
- Incorrect region coordinates
- Text visibility problems
"""

import sys
import os
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.verification_controller import VerificationController
from core.logger_config import setup_logging

def main():
    """Main debug function"""
    print("🔍 VLM Text Extraction Debug Tool")
    print("=" * 50)
    
    # Setup logging
    setup_logging(level=20)  # INFO level
    
    try:
        # Create verification controller
        controller = VerificationController()
        
        print("📸 Capturing current screen regions and asking VLM what it can read...")
        print("   This may take a few seconds depending on VLM model speed...")
        print()
        
        # Run VLM text extraction debug
        results = controller.debug_vlm_text_extraction()
        
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("✅ VLM Text Extraction Complete!")
        print()
        print("=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        
        for region_name, extracted_text in results.items():
            if region_name != "error":
                region_display = "📊 DATA ENTRY REGION" if region_name == "data_entry" else "📄 SOURCE DOCUMENT REGION"
                print(f"\n{region_display}:")
                print("-" * 40)
                print(extracted_text)
                print()
        
        print("=" * 60)
        print("💡 DEBUGGING TIPS:")
        print("- If text is missing, check region coordinates")
        print("- If text is garbled, try adjusting image enhancement settings")
        print("- If results are inconsistent, the VLM model may need fine-tuning")
        print("- Compare with OCR results to identify VLM vs OCR accuracy")
        
    except KeyboardInterrupt:
        print("\n🛑 Debug cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()