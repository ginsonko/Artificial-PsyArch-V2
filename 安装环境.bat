@echo off
setlocal EnableExtensions
chcp 65001 >nul

cd /d "%~dp0"

where py >nul 2>nul
if %errorlevel% neq 0 (
  echo [错误] 未找到 py 启动器，请先安装 Python 3.11+
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [信息] 正在创建 .venv ...
  py -3 -m venv .venv
  if %errorlevel% neq 0 (
    echo [错误] 创建虚拟环境失败
    pause
    exit /b %errorlevel%
  )
)

echo [信息] 正在升级 pip ...
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
  echo [错误] pip 升级失败
  pause
  exit /b %errorlevel%
)

echo [信息] 安装核心依赖 ...
.venv\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
  echo [错误] 核心依赖安装失败
  pause
  exit /b %errorlevel%
)

echo [信息] 安装可选增强依赖（视频 / FAISS）...
.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
if %errorlevel% neq 0 (
  echo [警告] 可选增强依赖未全部安装成功，核心能力仍可使用
)

echo [信息] 运行环境自检 ...
.venv\Scripts\python.exe scripts\doctor_env.py

echo.
echo [完成] 环境安装流程已结束
pause
endlocal
