@echo off
chcp 65001 >nul
echo ========================================
echo 汽车座舱RAG系统 - 启动诊断脚本
echo ========================================
echo.

echo [1/7] 检查Python环境...
python --version
if errorlevel 1 (
    echo [ERROR] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

echo.
echo [2/7] 检查依赖文件...
if not exist "requirements.txt" (
    echo [ERROR] 未找到requirements.txt文件
    pause
    exit /b 1
)

echo.
echo [3/7] 检查项目目录结构...
if not exist "data\raw" (
    echo [WARN] data\raw 目录不存在
    mkdir "data\raw"
    echo [OK] 已创建 data\raw
)

if not exist "data\chunks" (
    echo [WARN] data\chunks 目录不存在
    mkdir "data\chunks"
    echo [OK] 已创建 data\chunks
)

if not exist "vector_store\faiss_index" (
    echo [WARN] vector_store\faiss_index 目录不存在
    mkdir "vector_store\faiss_index"
    echo [OK] 已创建 vector_store\faiss_index
)

echo.
echo [4/7] 检查向量索引文件...
echo 检查目录: vector_store\faiss_index\

if exist "vector_store\faiss_index\faiss_index.bin" (
    echo   [FOUND] faiss_index.bin 存在
    if exist "vector_store\faiss_index\metadata.pkl" (
        echo   [FOUND] metadata.pkl 存在
        echo [OK] 找到完整的FAISS索引文件
    ) else (
        echo   [MISSING] metadata.pkl 不存在
        echo [WARN] 缺少元数据文件 (metadata.pkl)
    )
) else (
    echo   [MISSING] faiss_index.bin 不存在
    echo [WARN] 缺少FAISS索引文件 (faiss_index.bin)
)

echo.
echo [5/7] 检查Gradio应用...
if exist "gradio_web\app.py" (
    echo [OK] 找到Gradio应用文件
) else (
    echo [ERROR] 未找到Gradio应用文件
    pause
    exit /b 1
)

echo.
echo [6/7] 检查端口占用...
netstat -ano | findstr :7860 >nul
if %errorlevel% == 0 (
    echo [WARN] 端口7860已被占用
    echo 请关闭占用该端口的程序后重试
    echo 或者修改config.py中的端口号
) else (
    echo [OK] 端口7860可用
)

echo.
echo [7/7] 测试Python导入...
echo import gradio > test_import.py
echo import sys >> test_import.py
echo sys.path.append('.') >> test_import.py
echo from gradio_web.app import create_app >> test_import.py
echo print("[OK] 所有导入成功") >> test_import.py

python test_import.py
if errorlevel 1 (
    echo [ERROR] Python导入测试失败
    echo 请检查依赖包是否已安装
    del test_import.py
    pause
    exit /b 1
)
del test_import.py

echo.
echo ========================================
echo 诊断完成！
echo.
echo 启动方式：
echo   1. 双击运行 start.bat
echo   2. 命令行：python main.py
echo   3. 简化版：python main_simple.py
echo.
echo 访问地址：http://localhost:7860
echo ========================================
pause