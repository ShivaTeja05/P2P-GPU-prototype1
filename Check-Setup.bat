@echo off
setlocal
REM Verifies Docker, the NVIDIA runtime and Tailscale, then tests the GPU for real.
set "LOG=%~dp0check-setup-output.txt"
if not exist "%~dp0.venv\Scripts\p2pgpu.exe" (
  echo.
  echo   p2pgpu is not installed yet. Run Setup.bat first.
  echo.
  pause
  exit /b 1
)
echo Running checks, please wait...
echo.
"%~dp0.venv\Scripts\p2pgpu.exe" doctor %* > "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
type "%LOG%"
echo.
echo ============================================================
echo   Output was also saved to:
echo   %LOG%
echo   Send that file if you need help.
echo ============================================================
echo.
pause
exit /b %RC%
