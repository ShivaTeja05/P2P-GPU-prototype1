@echo off
REM First-time setup: installs prerequisites and p2pgpu itself.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" %*
pause
