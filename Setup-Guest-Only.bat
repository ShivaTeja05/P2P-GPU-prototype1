@echo off
REM Setup for using a friend's GPU without sharing your own (skips Docker).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup.ps1" -GuestOnly %*
pause
