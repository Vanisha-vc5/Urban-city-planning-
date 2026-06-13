@echo off
REM ============================================================
REM SmartCityAI — Dashboard Launcher
REM Double-click this file to start the dashboard
REM ============================================================

SET PYTHON=C:\Users\Vanisha\AppData\Local\Python\pythoncore-3.14-64\python.exe
SET SCRIPTS=C:\Users\Vanisha\AppData\Local\Python\pythoncore-3.14-64\Scripts
SET PROJECT_DIR=%~dp0

echo.
echo  ============================================
echo    SmartCityAI -- Urban Infrastructure AI
echo  ============================================
echo.
echo  Python:   %PYTHON%
echo  Project:  %PROJECT_DIR%
echo.

REM Add Scripts to PATH for this session
SET PATH=%SCRIPTS%;%PATH%

REM Move to project directory
cd /d "%PROJECT_DIR%"

echo  Checking installation...
%PYTHON% -c "import streamlit; print('  Streamlit OK:', streamlit.__version__)" 2>nul || (
    echo  Installing dependencies...
    %PYTHON% -m pip install -r requirements.txt --quiet
)

echo.
echo  Starting dashboard on http://localhost:8501
echo  Press Ctrl+C to stop.
echo.

%SCRIPTS%\streamlit.exe run dashboard\app.py --server.port 8501 --server.headless false

pause
