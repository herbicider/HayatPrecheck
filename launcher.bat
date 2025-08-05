@echo off
REM Quick launcher script for Pharmacy Verification tools

:menu
cls
echo Pharmacy Verification System - Tool Launcher
echo ============================================
echo.
echo Available tools:
echo 1. Main verification system (Precheck_OCR.py)
echo 2. Enhanced coordinate adjuster (coordinate_adjuster.py)
echo 3. Command-line coordinate helper (coordinate_helper.py)
echo 4. View current coordinates
echo 5. Validate coordinates
echo 6. Exit
echo.

set /p choice="Select tool (1-6): "

if "%choice%"=="1" (
    echo Starting main verification system...
    python Precheck_OCR.py
    goto continue
)

if "%choice%"=="2" (
    echo Starting enhanced coordinate adjuster...
    echo Note: Requires GUI support (tkinter)
    python coordinate_adjuster.py
    goto continue
)

if "%choice%"=="3" (
    echo Command-line coordinate helper usage:
    echo   show      - Display current coordinates
    echo   edit      - Edit coordinates for a field
    echo   validate  - Validate all coordinates
    echo   backup    - Create backup
    echo.
    set /p cmd="Enter command: "
    python coordinate_helper.py %cmd%
    goto continue
)

if "%choice%"=="4" (
    echo Current coordinates:
    python coordinate_helper.py show
    goto continue
)

if "%choice%"=="5" (
    echo Validating coordinates...
    python coordinate_helper.py validate
    goto continue
)

if "%choice%"=="6" (
    echo Goodbye!
    exit
)

echo Invalid choice. Please select 1-6.

:continue
echo.
pause
goto menu
