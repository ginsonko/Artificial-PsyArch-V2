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

set "DEVICE_INDEX=%~1"
if "%DEVICE_INDEX%"=="" set "DEVICE_INDEX=0"

%PY_CMD% -m observatory_v2 run-webcam-stream --device-index %DEVICE_INDEX% --max-frames 8 --label "BAT 摄像头流样例" --text-prefix "观察摄像头"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 摄像头流样例运行失败，错误码 %errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入新的摄像头流 run 结果到 outputs\runs\
pause
endlocal
