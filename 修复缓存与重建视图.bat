@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "PY_CMD="
if exist ".venv\Scripts\python.exe" (
  set "PY_CMD=.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    set "PY_CMD=py -3"
  ) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
      set "PY_CMD=python"
    )
  )
)

if "%PY_CMD%"=="" (
  echo [错误] 未找到 Python 3.11+。
  pause
  exit /b 1
)

echo ======================================
echo      重建 Live 视图与最新快照
echo ======================================
echo.

%PY_CMD% scripts\rebuild_live_preview.py

if %errorlevel% neq 0 (
  echo.
  echo [错误] 重建视图失败，错误码=%errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已输出当前最新 live 快照。
pause
endlocal

