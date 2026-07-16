@echo off
echo ========================================
echo Car Cockpit RAG System Startup Script
echo ========================================

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found, please install Python 3.8+
    pause
    exit /b 1
)

REM Check requirements
if not exist "requirements.txt" (
    echo Error: requirements.txt not found
    pause
    exit /b 1
)

echo.
echo 1. Checking dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Warning: Dependency installation failed, please install manually
)

echo.
echo 2. Checking directory structure...
if not exist "data\raw" mkdir data\raw
if not exist "data\chunks" mkdir data\chunks
if not exist "data\qa_train" mkdir data\qa_train
if not exist "vector_store\faiss_index" mkdir vector_store\faiss_index

echo.
echo 3. Starting Gradio Web interface...
echo Please visit: http://localhost:7860
echo.

python main.py

pause