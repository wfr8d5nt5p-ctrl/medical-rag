@echo off
chcp 65001 >nul
cd /d "%~dp0web"
set HF_ENDPOINT=https://hf-mirror.com
set PY="%~dp0.venv\Scripts\python.exe"
if not exist %PY% set PY=python
echo.
echo  ============================================
echo   正在启动 医知助手 网页版 ...
echo   启动后可保持本窗口最小化，不要关闭。
echo   访问地址: http://127.0.0.1:8000
echo  ============================================
echo.
start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:8000"
%PY% server.py
pause