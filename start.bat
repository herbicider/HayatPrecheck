@echo off
REM Optional launcher. Double-clicking app.pyw does the same thing, since
REM Windows associates .pyw with pythonw.exe (no console window).
REM
REM Use this file if you want a desktop shortcut or if .pyw is not associated.

cd /d "%~dp0"
start "" pythonw app.pyw
