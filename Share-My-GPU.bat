@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\share.ps1" %*
pause
