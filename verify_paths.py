#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证模型缓存路径
"""

from pathlib import Path

print("=" * 60)
print("验证模型缓存路径")
print("=" * 60)

paths_to_check = [
    ("F:\\ModelCache", "模型缓存根目录"),
    ("F:\\ModelCache\\huggingface", "HuggingFace目录"),
    ("F:\\ModelCache\\huggingface\\hub", "模型hub目录"),
    ("F:\\ModelCache\\huggingface\\datasets", "数据集缓存"),
    ("F:\\ModelCache\\huggingface\\modules", "模块缓存"),
]

for path_str, description in paths_to_check:
    path = Path(path_str)
    exists = path.exists()
    status = "[OK] 存在" if exists else "[ERROR] 不存在"
    print(f"{description}:")
    print(f"  路径: {path_str}")
    print(f"  状态: {status}")
    
    if exists and path.is_dir():
        items = list(path.iterdir())
        if items:
            print(f"  内容: {[item.name for item in items[:5]]}")
            if len(items) > 5:
                print(f"  更多... (共{len(items)}项)")
        else:
            print("  内容: (空目录)")
    print()

# 检查模型文件
print("检查模型文件:")
hub_path = Path("F:/ModelCache/huggingface/hub")
if hub_path.exists():
    models = [item for item in hub_path.iterdir() if item.is_dir() and item.name.startswith("models--")]
    print(f"  找到 {len(models)} 个模型:")
    
    required_models = {
        "Qwen/Qwen2.5-7B-Instruct": "models--Qwen--Qwen2.5-7B-Instruct",
        "BAAI/bge-small-zh-v1.5": "models--BAAI--bge-small-zh-v1.5",
        "bert-base-chinese": "models--bert-base-chinese",
        "openai/whisper-small": "models--openai--whisper-small"
    }
    
    for model_name, cache_name in required_models.items():
        model_path = hub_path / cache_name
        if model_path.exists():
            snapshots_dir = model_path / "snapshots"
            if snapshots_dir.exists():
                snapshots = list(snapshots_dir.iterdir())
                print(f"  [OK] {model_name}: 已缓存 ({len(snapshots)}个快照)")
            else:
                print(f"  [WARN] {model_name}: 目录存在但无快照")
        else:
            print(f"  [ERROR] {model_name}: 未找到")
else:
    print(f"  [ERROR] hub目录不存在: {hub_path}")

print("\n" + "=" * 60)
print("验证完成")
print("=" * 60)