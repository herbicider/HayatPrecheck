#!/bin/bash
# Quick launcher script for Pharmacy Verification tools

echo "Pharmacy Verification System - Tool Launcher"
echo "============================================"
echo ""
echo "Available tools:"
echo "1. Main verification system (Precheck_OCR.py)"
echo "2. Enhanced coordinate adjuster (coordinate_adjuster.py)"
echo "3. Command-line coordinate helper (coordinate_helper.py)"
echo "4. View current coordinates"
echo "5. Validate coordinates"
echo "6. Exit"
echo ""

while true; do
    read -p "Select tool (1-6): " choice
    
    case $choice in
        1)
            echo "Starting main verification system..."
            python3 Precheck_OCR.py
            ;;
        2)
            echo "Starting enhanced coordinate adjuster..."
            echo "Note: Requires GUI support (tkinter)"
            python3 coordinate_adjuster.py
            ;;
        3)
            echo "Command-line coordinate helper usage:"
            echo "  show      - Display current coordinates"
            echo "  edit      - Edit coordinates for a field"
            echo "  validate  - Validate all coordinates"
            echo "  backup    - Create backup"
            echo ""
            read -p "Enter command: " cmd
            python3 coordinate_helper.py $cmd
            ;;
        4)
            echo "Current coordinates:"
            python3 coordinate_helper.py show
            ;;
        5)
            echo "Validating coordinates..."
            python3 coordinate_helper.py validate
            ;;
        6)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please select 1-6."
            ;;
    esac
    
    echo ""
    echo "Press Enter to continue..."
    read
    echo ""
done
