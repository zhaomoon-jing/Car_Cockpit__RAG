@echo off
chcp 65001 >nul
echo ========================================
echo 汽车座舱RAG系统启动脚本
echo ========================================

REM 检查Python
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖
if not exist "requirements.txt" (
    echo 错误: 未找到requirements.txt文件
    pause
    exit /b 1
)

echo.
echo 1. 检查依赖包...
pip install -r requirements.txt
if errorlevel 1 (
    echo 警告: 依赖安装失败，请手动安装
)

echo.
echo 2. 检查数据目录结构...
if not exist "data\raw" mkdir data\raw
if not exist "data\chunks" mkdir data\chunks
if not exist "data\qa_train" mkdir data\qa_train
if not exist "vector_store\faiss_index" mkdir vector_store\faiss_index

echo.
echo 3. 启动Gradio Web界面...
echo 请访问: http://localhost:7860
echo.

python main.py

pause