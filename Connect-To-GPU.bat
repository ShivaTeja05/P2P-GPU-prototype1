@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\connect.ps1" %*
pause
