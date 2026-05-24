@echo off
setlocal EnableExtensions
chcp 65001 >nul

cd /d "%~dp0"

"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0launchers\start-observatory.ps1"
set "EXIT_CODE=%errorlevel%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] Failed to start observatory. Exit code=%EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

endlocal
