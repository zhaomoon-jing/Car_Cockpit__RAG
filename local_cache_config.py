#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地模型缓存配置 - 使用已下载的HuggingFace模型缓存
"""

import os
from pathlib import Path

# 设置HuggingFace缓存目录（使用自定义路径）
HF_CACHE_DIR = Path("F:/ModelCache/huggingface/hub")

# 检查缓存目录是否存在
if not HF_CACHE_DIR.exists():
    print(f"❌ HuggingFace缓存目录不存在: {HF_CACHE_DIR}")
    print("请确保模型已下载到该目录")
    exit(1)

print(f"✅ 找到HuggingFace缓存目录: {HF_CACHE_DIR}")

# 检查各个模型目录
models_to_check = {
    "llm": "models--Qwen--Qwen3-0.6B-Instruct",
    "embedding": "models--BAAI--bge-small-zh-v1.5",
    "intent": "models--bert-base-chinese",
    "asr": "models--openai--whisper-small"
}

print("\n🔍 检查已下载的模型:")
for model_type, model_dir in models_to_check.items():
    model_path = HF_CACHE_DIR / model_dir
    if model_path.exists():
        print(f"  ✅ {model_type}: {model_dir}")
        
        # 检查snapshots目录
        snapshots_dir = model_path / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                print(f"      快照: {len(snapshots)} 个")
                for snap in snapshots[:3]:  # 显示前3个快照
                    print(f"        - {snap.name}")
    else:
        print(f"  ⚠️  {model_type}: {model_dir} (未找到)")

# 配置使用本地缓存的环境变量
print("\n📋 配置本地缓存使用:")
print("设置以下环境变量来使用本地缓存:")

env_config = {
    "HF_HOME": str(HF_CACHE_DIR.parent),
    "TRANSFORMERS_CACHE": str(HF_CACHE_DIR),
    "HF_DATASETS_CACHE": str(HF_CACHE_DIR.parent / "datasets"),
    "HF_MODULES_CACHE": str(HF_CACHE_DIR.parent / "modules"),
    "HF_HUB_OFFLINE": "1",  # 离线模式，强制使用本地缓存
}

for key, value in env_config.items():
    print(f"export {key}={value}")

# 生成Windows批处理文件
print("\n🖥️ Windows批处理文件 (start_local.bat):")
bat_content = """@echo off
chcp 65001 >nul

echo ========================================
echo 汽车座舱RAG系统 - 本地缓存模式启动脚本
echo ========================================
echo.

echo [1/3] 设置环境变量...
set HF_HOME=F:\ModelCache\huggingface
set TRANSFORMERS_CACHE=F:\ModelCache\huggingface\hub
set HF_DATASETS_CACHE=F:\ModelCache\huggingface\datasets
set HF_MODULES_CACHE=F:\ModelCache\huggingface\modules
set HF_HUB_OFFLINE=1
echo ✅ 环境变量设置完成

echo.
echo [2/3] 检查本地模型缓存...
python -c "
import os
from pathlib import Path

cache_dir = Path(r'F:\\ModelCache\\huggingface\\hub')
models = {
    'LLM模型': 'models--Qwen--Qwen2.5-7B-Instruct',
    '嵌入模型': 'models--BAAI--bge-small-zh-v1.5',
}

print('📁 检查本地模型缓存:')
for name, model_dir in models.items():
    model_path = cache_dir / model_dir
    if model_path.exists():
        snapshots = list((model_path / 'snapshots').iterdir())
        if snapshots:
            print(f'  ✅ {name}: {model_dir} ({len(snapshots)}个快照)')
        else:
            print(f'  ⚠️ {name}: {model_dir} (目录为空)')
    else:
        print(f'  ❌ {name}: {model_dir} (未找到)')
"
if errorlevel 1 (
    echo ❌ 检查本地模型缓存失败
    pause
    exit /b 1
)

echo.
echo [3/3] 启动应用（离线模式）...
echo ⚠️ 注意：使用本地缓存模型，无需网络下载
echo.

python main.py

echo.
echo ========================================
echo 应用已关闭
echo ========================================
pause
"""

# 保存批处理文件
bat_path = Path(__file__).parent / "start_local.bat"
with open(bat_path, "w", encoding="utf-8") as f:
    f.write(bat_content)
    
print(f"✅ 批处理文件已保存到: {bat_path}")

# 创建Python脚本使用本地缓存
print("\n🐍 Python配置脚本 (use_local_cache.py):")
py_content = """#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用本地HuggingFace缓存的配置脚本
"""

import os
import sys
from pathlib import Path

def setup_local_cache():
    \"\"\"设置本地缓存环境\"\"\"
    
    # HuggingFace缓存目录
    hf_cache_dir = Path(r"F:\\ModelCache\\huggingface")
    hub_dir = hf_cache_dir / "hub"
    
    if not hub_dir.exists():
        print(f"❌ HuggingFace缓存目录不存在: {hub_dir}")
        return False
    
    # 设置环境变量
    os.environ["HF_HOME"] = str(hf_cache_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hub_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache_dir / "datasets")
    os.environ["HF_MODULES_CACHE"] = str(hf_cache_dir / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"  # 离线模式
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    print("✅ 本地缓存环境已设置:")
    print(f"   HF_HOME: {os.environ['HF_HOME']}")
    print(f"   TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    print(f"   HF_HUB_OFFLINE: {os.environ['HF_HUB_OFFLINE']}")
    
    return True

def check_local_models():
    \"\"\"检查本地缓存中的模型\"\"\"
    
    hub_dir = Path(os.environ.get("TRANSFORMERS_CACHE", ""))
    if not hub_dir.exists():
        print("❌ TRANSFORMERS_CACHE目录不存在")
        return
    
    print("\\n🔍 检查本地模型:")
    
    models = {
        "LLM模型 (Qwen/Qwen2.5-7B-Instruct)": "models--Qwen--Qwen2.5-7B-Instruct",
        "嵌入模型 (BAAI/bge-small-zh-v1.5)": "models--BAAI--bge-small-zh-v1.5",
    }
    
    all_available = True
    for name, model_dir in models.items():
        model_path = hub_dir / model_dir
        if model_path.exists():
            snapshots_dir = model_path / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                if snapshots:
                    print(f"  ✅ {name}: 可用 ({len(snapshots)}个快照)")
                    # 显示最新快照路径
                    latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                    print(f"      最新快照: {latest_snapshot}")
                else:
                    print(f"  ⚠️ {name}: 目录为空")
                    all_available = False
            else:
                print(f"  ⚠️ {name}: 无快照目录")
                all_available = False
        else:
            print(f"  ❌ {name}: 未找到")
            all_available = False
    
    return all_available

def get_model_local_path(model_name: str) -> str:
    \"\"\"获取模型的本地缓存路径\"\"\"
    
    # 将模型名称转换为缓存目录格式
    # 例如: Qwen/Qwen2.5-7B-Instruct -> models--Qwen--Qwen2.5-7B-Instruct
    cache_dir_name = model_name.replace("/", "--")
    if not cache_dir_name.startswith("models--"):
        cache_dir_name = f"models--{cache_dir_name}"
    
    hub_dir = Path(os.environ.get("TRANSFORMERS_CACHE", ""))
    model_cache_dir = hub_dir / cache_dir_name
    
    if not model_cache_dir.exists():
        return None
    
    snapshots_dir = model_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    
    snapshots = list(snapshots_dir.iterdir())
    if not snapshots:
        return None
    
    # 使用最新的快照
    latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
    return str(latest_snapshot)

if __name__ == "__main__":
    print("=" * 60)
    print("本地HuggingFace缓存配置工具")
    print("=" * 60)
    
    # 设置环境
    if not setup_local_cache():
        sys.exit(1)
    
    # 检查模型
    if check_local_models():
        print("\\n🎉 所有必需模型都已在本地缓存中可用！")
        print("\\n📝 使用说明:")
        print("1. 运行此脚本设置环境: python use_local_cache.py")
        print("2. 启动应用: python main.py")
        print("\\n💡 或者使用批处理文件: start_local.bat")
    else:
        print("\\n⚠️  部分模型未找到，可能需要下载")
        print("\\n📥 下载缺失模型:")
        print("1. 临时关闭离线模式: set HF_HUB_OFFLINE=0")
        print("2. 运行: python download_models.py")
        print("3. 重新启用离线模式: set HF_HUB_OFFLINE=1")
"""

py_path = Path(__file__).parent / "use_local_cache.py"
with open(py_path, "w", encoding="utf-8") as f:
    f.write(py_content)
    
print(f"✅ Python脚本已保存到: {py_path}")

print("\n" + "="*60)
print("🎉 配置完成！")
print("="*60)
print("\n📋 下一步:")
print("1. 运行: python use_local_cache.py 设置本地缓存环境")
print("2. 或双击: start_local.bat 启动离线模式")
print("3. 或直接使用: python main.py (已自动检测本地缓存)")
print("\n💡 优势:")
print("   - ✅ 无需网络连接")
print("   - ✅ 启动速度更快")
print("   - ✅ 避免重复下载")
print("   - ✅ 节省带宽和流量")
print("\n⚠️  注意:")
print("   - 确保模型已完整下载到缓存目录")
print("   - 如果需要更新模型，请先禁用离线模式")
print("   - 支持所有已下载的HuggingFace模型")