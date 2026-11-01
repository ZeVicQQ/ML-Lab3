@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================
echo   Running Lab 3 (Linear Classification)
echo ============================================

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

echo Checking dependencies...
".venv\Scripts\python.exe" -c "import numpy, sklearn, matplotlib, pandas" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies from requirements.txt...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies. Check requirements.txt
        pause
        exit /b 1
    )
)

echo.
echo Running lab3.py ...
".venv\Scripts\python.exe" lab3.py

echo.
echo Done. Check lab3_results folder.
pause