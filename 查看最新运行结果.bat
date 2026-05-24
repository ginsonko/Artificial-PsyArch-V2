@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist "outputs\runs" (
  echo [提示] 当前还没有运行结果目录。
  pause
  exit /b 0
)

echo 正在打开最新运行目录...
start "" explorer.exe "%cd%\outputs\runs"
endlocal

