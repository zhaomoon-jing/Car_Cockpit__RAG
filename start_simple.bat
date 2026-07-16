@echo off
chcp 65001 >nul
echo ========================================
echo 汽车座舱RAG系统 - 简化启动脚本
echo ========================================
echo.

echo [1/5] 检查Python环境...
python --version
if errorlevel 1 (
    echo ❌ 未找到Python，请安装Python 3.8+
    pause
    exit /b 1
)

echo ✅ Python环境正常

echo.
echo [2/5] 检查依赖包...
python -c "import transformers" 2>nul
if errorlevel 1 (
    echo ⚠️ transformers未安装，正在安装...
    pip install transformers==4.44.2
    if errorlevel 1 (
        echo ❌ transformers安装失败
        echo 请手动安装: pip install transformers==4.44.2
        pause
        exit /b 1
    )
)



python -c "import gradio" 2>nul
if errorlevel 1 (
    echo  gradio version
    pip show gradio
)

echo ✅ 依赖包检查完成

echo.
echo [3/5] 设置环境变量...
set HF_ENDPOINT=https://hf-mirror.com
set TRANSFORMERS_OFFLINE=0
set HF_HUB_DOWNLOAD_TIMEOUT=120

echo ✅ 环境变量设置完成

echo.
echo [4/5] 检查数据目录...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\chunks" mkdir "data\chunks"
if not exist "data\qa_train" mkdir "data\qa_train"
if not exist "vector_store\faiss_index" mkdir "vector_store\faiss_index"
if not exist "logs" mkdir "logs"

echo ✅ 目录结构检查完成

echo.
echo [5/5] 启动应用...
echo.
echo ⚠️ 注意：首次运行会下载模型文件，可能需要较长时间
echo ⚠️ 如果下载失败，请运行：python download_models.py
echo.

python main_simple.py

echo.
echo ========================================
echo 应用已关闭
echo ========================================
pause