@echo off
setlocal
REM Downloads the ~4 GB session image now, so starting a share later is instant.
if not exist "%~dp0.venv\Scripts\p2pgpu.exe" (
  echo.
  echo   p2pgpu is not installed yet. Run Setup.bat first.
  echo.
  pause
  exit /b 1
)
echo Downloading the session image. This is a few GB and only happens once.
echo You can keep using your computer while it runs.
echo.
"%~dp0.venv\Scripts\p2pgpu.exe" prepare %*
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
