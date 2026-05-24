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
echo      AP 二期 Phase1 批量演示运行
echo ======================================
echo.

%PY_CMD% scripts\run_phase1_batch_demo.py --runs 3 --ticks 8 --interval-ms 30

if %errorlevel% neq 0 (
  echo.
  echo [错误] 批量演示运行失败，错误码=%errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入 outputs\runs\ 下的多次最小演示运行。
pause
endlocal

