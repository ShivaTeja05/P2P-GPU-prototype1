@echo off
REM Verifies Docker, the NVIDIA runtime and Tailscale, then tests the GPU for real.
"%~dp0.venv\Scripts\p2pgpu.exe" doctor %*
pause
