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

%PY_CMD% -m observatory_v2 run-autonomous --ticks 6 --text-hint "观察当前桌面并尝试持续聚焦" --stop-on-capture-failures 3 --stop-on-action-errors 4 --stop-on-idle-ticks 6 --idle-backoff-ms 150 --label "BAT 自主循环样例"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 自主循环样例失败，错误码 %errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入新的自主循环 run 结果到 outputs\runs\
pause
endlocal
