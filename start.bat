@echo off
echo Pharmacy Verification System
echo ===============================
echo.
echo Clearing Python cache...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc >nul 2>&1
echo Cache cleared!
echo.
echo Starting the launcher...
python launcher.py
pause
