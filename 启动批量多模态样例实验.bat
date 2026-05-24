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

echo ======================================
echo      AP 二期批量多模态样例实验
echo ======================================
echo.

%PY_CMD% -m observatory_v2 run-dataset --dataset config\sample_dataset_multimodal.json --label "BAT 触发的 Phase15 批量多模态样例实验"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 批量多模态样例实验失败，错误码=%errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入 outputs\runs\ 下的新 batch run。
pause
endlocal
