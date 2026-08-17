@echo off
setlocal
REM Lets the session container reach the cluster coordinator.
REM Needed on Windows: the container's host is the WSL2 VM, which does not carry
REM the Windows Tailscale interface, so the container cannot reach another
REM machine's 100.x address on its own. Measured, not assumed.
if not exist "%~dp0.venv\Scripts\p2pgpu.exe" (
  echo.
  echo   p2pgpu is not installed yet. Run Setup.bat first.
  echo.
  pause
  exit /b 1
)

set "COORD=%~1"
if "%COORD%"=="" (
  echo.
  echo   Paste the coordinator URL ^(the person running the cluster sends it^).
  echo   It looks like:  http://100.65.244.36:8899
  echo.
  set /p COORD="   Coordinator URL: "
)
if "%COORD%"=="" (
  echo   No URL given, nothing to do.
  pause
  exit /b 1
)

echo.
echo Starting the relay. Leave this window open for the whole session.
echo Windows may ask to allow it through the firewall - say yes, private networks.
echo.
"%~dp0.venv\Scripts\p2pgpu.exe" cluster relay --coordinator "%COORD%"
set "RC=%ERRORLEVEL%"
echo.
pause
exit /b %RC%
