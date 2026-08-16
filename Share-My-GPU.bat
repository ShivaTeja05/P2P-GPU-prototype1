@echo off
setlocal
if not exist "%~dp0.venv\Scripts\p2pgpu.exe" (
  echo.
  echo   p2pgpu is not installed yet. Run Setup.bat first.
  echo.
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\share.ps1" %*
echo.
pause
