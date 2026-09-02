@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo  CNC Tool Wear Prediction and Diagnosis System
echo ==============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo Install Python 3 and add it to PATH, then retry.
    pause
    exit /b 1
)

echo [1/2] Starting backend (FastAPI) ...
start "Backend" cmd /k "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"

echo Waiting for backend to warm up ...
timeout /t 5 /nobreak >nul

echo [2/2] Starting frontend (Streamlit) ...
start "Frontend" cmd /k "python -m streamlit run web/app.py"

echo.
echo Servers are starting in two separate windows.
echo   Backend  : http://127.0.0.1:8000/docs
echo   Frontend : http://localhost:8501
echo.
echo Keep both server windows open while using the app.
echo To stop a server, close its window or press Ctrl+C inside it.
echo.
pause
endlocal
