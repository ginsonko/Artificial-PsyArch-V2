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
  echo [错误] 未找到 Python 3.11+
  pause
  exit /b 1
)

set "CKPT=outputs\runtime_checkpoint.json"
if not exist "%CKPT%" (
  echo [错误] 未找到 checkpoint: %CKPT%
  echo 请先双击“导出当前Runtime检查点.bat”
  pause
  exit /b 1
)

echo ======================================
echo   从 checkpoint 继续文本运行
echo ======================================
echo 使用 checkpoint: %CKPT%
echo.

%PY_CMD% -m observatory_v2 continue-from-checkpoint --in "%CKPT%" --text "今天 天气 有点 冷" --text "算了 不说了" --label "BAT 继续运行"

if %errorlevel% neq 0 (
  echo.
  echo [错误] 继续运行失败，错误码=%errorlevel%
  pause
  exit /b %errorlevel%
)

echo.
echo [完成] 已从 checkpoint 继续运行并写入新的 outputs\runs\ 结果
pause
endlocal
