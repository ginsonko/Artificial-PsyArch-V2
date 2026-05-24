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
echo        AP 二期截图感知实验
echo ======================================
echo 说明:
echo 1. 默认只运行 1 个截图 tick
echo 2. 若配置里未开启截图采样，则会进入安全空跑模式
echo.

%PY_CMD% -m observatory_v2 run-screen --ticks 1 --text "观测台截图感知样例" --label "BAT 截图感知运行"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 截图感知实验失败，错误码 %errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入新的截图感知 run 结果到 outputs\runs\
pause
endlocal
