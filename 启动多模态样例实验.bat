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
echo        AP 二期多模态样例实验
echo ======================================
echo 当前版本会使用 repo 内置的图片/音频内联样例
echo 并走统一多模态 run 主链
echo.

%PY_CMD% scripts\batch_runner_v2.py --dataset config\sample_dataset_multimodal.json --label "BAT 多模态样例运行"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 多模态样例运行失败，错误码 %errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入新的多模态 run 结果到 outputs\runs\
pause
endlocal
