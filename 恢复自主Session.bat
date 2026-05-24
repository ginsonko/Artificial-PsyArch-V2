@echo off
setlocal EnableExtensions
chcp 65001 >nul

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
  echo [错误] 未找到 Python 3.11+
  pause
  exit /b 1
)

if "%~1"=="" (
  %PY_CMD% -m observatory_v2 recover-autonomous-session
) else (
  %PY_CMD% -m observatory_v2 recover-autonomous-session --run-id "%~1"
)

if %errorlevel% neq 0 (
  echo.
  echo [错误] 恢复自主 Session 失败，错误码 %errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已触发恢复自主 Session 请求
pause
endlocal
