@echo off
chcp 65001 >nul
echo ========================================
echo 汽车座舱RAG系统 - 改进启动脚本
echo ========================================
echo.

echo [1/6] 检查Python环境...
python --version
if errorlevel 1 (
    echo ❌ 未找到Python，请安装Python 3.8+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python环境正常

echo.
echo [2/6] 设置环境变量...
set HF_ENDPOINT=https://hf-mirror.com
set HF_HUB_DOWNLOAD_TIMEOUT=120
set HTTP_PROXY=http://127.0.0.1:7890
set HTTPS_PROXY=http://127.0.0.1:7890
echo ✅ 环境变量设置完成

echo.
echo [3/6] 检查依赖包...
echo 正在检查transformers...
python -c "import transformers" 2>nul
if errorlevel 1 (
    echo ⚠️ transformers未安装，正在安装...
    pip install transformers==4.36.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo ❌ transformers安装失败
        echo 请尝试: pip install transformers
        pause
        exit /b 1
    )
)

echo 正在检查torch...
python -c "import torch" 2>nul
if errorlevel 1 (
    echo ⚠️ torch未安装，正在安装...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo ❌ torch安装失败
        echo 请尝试: pip install torch
        pause
        exit /b 1
    )
)

echo 正在检查gradio...
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo ⚠️ gradio未安装，正在安装...
    pip install gradio==4.19.0 --index-url https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo ❌ gradio安装失败
        echo 请尝试: pip install gradio
        pause
        exit /b 1
    )
)

echo ✅ 依赖包检查完成

echo.
echo [4/6] 检查数据目录...
if not exist "data\raw" mkdir "data\raw"
if not exist "data\chunks" mkdir "data\chunks"
if not exist "data\qa_train" mkdir "data\qa_train"
if not exist "vector_store\faiss_index" mkdir "vector_store\faiss_index"
if not exist "logs" mkdir "logs"
if not exist "F:\ModelCache" (
    echo ⚠️ 模型缓存目录 F:\ModelCache 不存在，将在项目内创建
    if not exist "models_cache" mkdir "models_cache"
)

echo ✅ 目录结构检查完成

echo.
echo [5/6] 测试网络连接...
echo 正在测试HuggingFace连接...
python -c "
import requests
import sys

mirrors = [
    ('HF Mirror', 'https://hf-mirror.com'),
    ('HuggingFace', 'https://huggingface.co'),
    ('Google', 'https://www.google.com')
]

print('网络连接测试:')
all_ok = True
for name, url in mirrors:
    try:
        response = requests.get(url, timeout=10)
        print(f'  ✅ {name}: 可访问 (状态码: {response.status_code})')
    except Exception as e:
        print(f'  ❌ {name}: 不可访问 - {str(e)[:50]}')
        all_ok = False

if not all_ok:
    print('\\n⚠️ 网络连接有问题，请检查网络或设置代理')
    print('   设置代理: set HTTP_PROXY=http://127.0.0.1:7890')
    sys.exit(1)
else:
    print('\\n✅ 网络连接正常')
" 2>nul
if errorlevel 1 (
    echo ❌ 网络连接测试失败
    echo.
    echo 💡 解决方案：
    echo     1. 检查网络连接
    echo     2. 设置代理：set HTTP_PROXY=http://127.0.0.1:7890
    echo     3. 使用离线模式：python offline_config.py
    echo     4. 运行简化版本：start_simple.bat
    echo.
    choice /c YN /m "是否继续启动？（可能下载失败）"
    if errorlevel 2 (
        echo 用户取消启动
        pause
        exit /b 0
    )
)

echo.
echo [6/6] 启动应用...
echo.
echo ⚠️ 重要提示：
echo     1. 首次启动需要下载模型（约10-20分钟）
echo     2. 如果下载失败，程序会自动尝试备用模型
echo     3. 请保持网络连接稳定
echo.
echo 💡 如果遇到连接超时错误：
echo     1. 按 Ctrl+C 中断程序
echo     2. 运行: python download_models.py （选择小模型）
echo     3. 或运行: python test_model_load.py （测试连接）
echo     4. 使用离线模式: python offline_config.py
echo.
echo 🔗 启动中，请稍候...
echo.

python main.py

if errorlevel 1 (
    echo.
    echo ❌ 启动失败！
    echo.
    echo 🔧 故障排除：
    echo     1. 运行测试：python test_model_load.py
    echo     2. 下载模型：python download_models.py
    echo     3. 使用离线：python offline_config.py
    echo     4. 简化启动：start_simple.bat
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 应用已关闭
echo ========================================
pause