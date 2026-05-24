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
echo      AP 二期单次实验
echo ======================================
echo.

%PY_CMD% -m observatory_v2 run-text --text "今天 天气 不错" --text "今天 天气 不错" --text "我 想 出门" --label "BAT 触发的 Phase2 文本最小闭环运行"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 单次实验执行失败，错误码=%errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已写入 outputs\runs\ 下的最新文本闭环 run。
pause
endlocal
