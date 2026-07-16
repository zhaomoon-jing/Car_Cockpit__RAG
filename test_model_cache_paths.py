#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型缓存路径配置
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("测试模型缓存路径配置")
print("=" * 60)

# 检查F:\ModelCache目录
model_cache_path = Path("F:/ModelCache")
print(f"\n1. 检查F:\\ModelCache目录:")
print(f"   路径: {model_cache_path}")
print(f"   存在: {model_cache_path.exists()}")
if model_cache_path.exists():
    print(f"   内容: {[item.name for item in model_cache_path.iterdir() if item.is_dir()]}")

# 检查HuggingFace目录结构
hf_path = Path("F:/ModelCache/huggingface")
print(f"\n2. 检查HuggingFace目录结构:")
print(f"   路径: {hf_path}")
print(f"   存在: {hf_path.exists()}")
if hf_path.exists():
    print(f"   内容: {[item.name for item in hf_path.iterdir() if item.is_dir()]}")

# 检查hub目录
hub_path = Path("F:/ModelCache/huggingface/hub")
print(f"\n3. 检查hub目录:")
print(f"   路径: {hub_path}")
print(f"   存在: {hub_path.exists()}")
if hub_path.exists():
    models = [item.name for item in hub_path.iterdir() if item.is_dir() and item.name.startswith("models--")]
    print(f"   模型数量: {len(models)}")
    if models:
        print(f"   模型列表: {models[:10]}")  # 只显示前10个

# 检查配置文件中的路径
print(f"\n4. 检查配置文件路径:")
try:
    from config import MODEL_CONFIG
    cache_path = MODEL_CONFIG.get("model_download", {}).get("local_cache_path", "")
    print(f"   config.py 缓存路径: {cache_path}")
    print(f"   路径存在: {Path(cache_path).exists() if cache_path else '未设置'}")
except ImportError as e:
    print(f"   无法导入config.py: {e}")

# 检查环境变量
print(f"\n5. 检查环境变量:")
env_vars = ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "HF_MODULES_CACHE"]
for var in env_vars:
    value = os.environ.get(var)
    print(f"   {var}: {value}")
    if value:
        print(f"     路径存在: {Path(value).exists()}")

# 测试从缓存加载模型
print(f"\n6. 测试模型加载:")
try:
    # 尝试导入transformers
    from transformers import AutoTokenizer, AutoModel
    print("   ✅ transformers 已安装")
    
    # 设置环境变量
    os.environ["TRANSFORMERS_CACHE"] = "F:/ModelCache/huggingface/hub"
    os.environ["HF_HOME"] = "F:/ModelCache/huggingface"
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # 检查模型是否存在
    model_names = [
        "Qwen/Qwen2.5-7B-Instruct",
        "BAAI/bge-small-zh-v1.5",
        "bert-base-chinese",
        "openai/whisper-small"
    ]
    
    for model_name in model_names:
        cache_name = model_name.replace("/", "--")
        if not cache_name.startswith("models--"):
            cache_name = f"models--{cache_name}"
        
        model_path = hub_path / cache_name
        if model_path.exists():
            snapshots = list((model_path / "snapshots").iterdir()) if (model_path / "snapshots").exists() else []
            print(f"   ✅ {model_name}: 已缓存 ({len(snapshots)}个快照)")
        else:
            print(f"   ❌ {model_name}: 未找到")
            
except ImportError as e:
    print(f"   ❌ transformers 未安装: {e}")
except Exception as e:
    print(f"   ❌ 测试失败: {e}")

print(f"\n" + "=" * 60)
print("测试完成")
print("=" * 60)