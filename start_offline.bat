@echo off
chcp 65001 >nul

echo ========================================
echo 汽车座舱RAG系统 - 本地缓存离线模式
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
echo [2/6] 设置本地缓存环境...
set HF_HOME=F:\ModelCache\huggingface
set TRANSFORMERS_CACHE=F:\ModelCache\huggingface\hub
set HF_DATASETS_CACHE=F:\ModelCache\huggingface\datasets
set HF_MODULES_CACHE=F:\ModelCache\huggingface\modules
set HF_HUB_OFFLINE=1
echo ✅ 环境变量设置完成

echo.
echo [3/6] 检查本地模型缓存...
if not exist "F:\ModelCache\huggingface\hub" (
    echo ❌ HuggingFace缓存目录不存在: F:\ModelCache\huggingface\hub
    echo.
    echo 📥 请先下载模型：
    echo     1. 运行: python download_models.py
    echo     2. 或运行: python fix_huggingface.py
    echo.
    choice /c YN /m "是否立即下载模型？"
    if errorlevel 2 (
        echo 用户取消启动
        pause
        exit /b 0
    )
    echo.
    echo 📥 正在下载模型...
    python download_models.py
    if errorlevel 1 (
        echo ❌ 模型下载失败
        pause
        exit /b 1
    )
)

echo.
echo [4/6] 验证模型完整性...
python -c "
import sys
from pathlib import Path

cache_dir = Path(r'F:\\ModelCache\\huggingface\\hub')
required_models = {
    'LLM模型': 'models--Qwen--Qwen2.5-7B-Instruct',
    '嵌入模型': 'models--BAAI--bge-small-zh-v1.5',
}

print('🔍 检查必需模型:')
all_found = True
for name, model_dir in required_models.items():
    model_path = cache_dir / model_dir
    if model_path.exists():
        snapshots = list((model_path / 'snapshots').iterdir())
        if snapshots:
            print(f'  ✅ {name}: 已下载 ({len(snapshots)}个快照)')
        else:
            print(f'  ⚠️ {name}: 目录存在但无快照')
            all_found = False
    else:
        print(f'  ❌ {name}: 未找到')
        all_found = False

if not all_found:
    print('\\n⚠️ 警告：部分模型未找到或未完整下载')
    sys.exit(1)
"
if errorlevel 1 (
    echo ❌ 必需模型不完整
    echo.
    echo 📥 请下载缺失的模型：
    echo     运行: python download_models.py
    echo.
    pause
    exit /b 1
)

echo ✅ 模型验证通过

echo.
echo [5/6] 检查依赖包...
echo 正在检查transformers...
python -c "import transformers" 2>nul
if errorlevel 1 (
    echo ⚠️ transformers未安装，正在安装...
    pip install transformers==4.36.0
    if errorlevel 1 (
        echo ❌ transformers安装失败
        echo 请尝试: pip install transformers --index-url https://pypi.tuna.tsinghua.edu.cn/simple
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
        echo 请尝试: pip install torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple
        pause
        exit /b 1
    )
)

echo 正在检查sentence-transformers...
python -c "import sentence_transformers" 2>nul
if errorlevel 1 (
    echo ⚠️ sentence-transformers未安装，正在安装...
    pip install sentence-transformers
    if errorlevel 1 (
        echo ❌ sentence-transformers安装失败
        echo 请尝试: pip install sentence-transformers --index-url https://pypi.tuna.tsinghua.edu.cn/simple
        pause
        exit /b 1
    )
)

echo 正在检查gradio...
python -c "import gradio" 2>nul
if errorlevel 1 (
    echo ⚠️ gradio未安装，正在安装...
    pip install gradio==4.19.0
    if errorlevel 1 (
        echo ❌ gradio安装失败
        echo 请尝试: pip install gradio --index-url https://pypi.tuna.tsinghua.edu.cn/simple
        pause
        exit /b 1
    )
)

echo ✅ 依赖包检查完成

echo.
echo [6/6] 启动应用（离线模式）...
echo ⚠️ 注意：使用本地缓存模型，无需网络连接
echo 📊 模型已从本地加载，启动速度更快
echo.

python main.py

if errorlevel 1 (
    echo.
    echo ❌ 启动失败！
    echo.
    echo 🔧 故障排除：
    echo     1. 检查缓存目录：F:\ModelCache\huggingface\hub
    echo     2. 运行测试：python test_model_load.py
    echo     3. 重新下载模型：python download_models.py
    echo     4. 使用在线模式：set HF_HUB_OFFLINE=0 && python main.py
    echo     5. 使用简化版本：python main_simple.py
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo 应用已关闭
echo ========================================
pause