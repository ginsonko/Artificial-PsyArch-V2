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
  echo 用法: 启动视频流样例实验.bat 你的视频文件.mp4
  pause
  exit /b 1
)

%PY_CMD% -m observatory_v2 run-video-stream --video "%~1" --text-prefix "视频流样例" --tick-fps 4 --label "BAT 视频流样例运行"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 视频流样例运行失败，错误码 %errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入新的视频流 run 结果到 outputs\runs\
pause
endlocal
