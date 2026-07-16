#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试本地缓存功能
"""

import os
import sys
from pathlib import Path

def main():
    print("=" * 60)
    print("测试本地缓存功能")
    print("=" * 60)
    
    # 设置环境变量
    cache_dir = Path("F:/ModelCache/huggingface/hub")
    
    print(f"1. 检查缓存目录: {cache_dir}")
    if cache_dir.exists():
        print("   [OK] 缓存目录存在")
        
        # 列出模型
        print(f"\n2. 缓存目录中的模型:")
        model_count = 0
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                model_name = item.name[8:].replace("--", "/")
                snapshots_dir = item / "snapshots"
                snapshot_count = len(list(snapshots_dir.iterdir())) if snapshots_dir.exists() else 0
                print(f"   - {model_name}: {snapshot_count} 个快照")
                model_count += 1
        
        print(f"\n   总共找到 {model_count} 个模型")
        
        # 检查必需模型
        print("\n3. 检查必需模型:")
        required_models = [
            "models--Qwen--Qwen2.5-7B-Instruct",
            "models--BAAI--bge-small-zh-v1.5"
        ]
        
        all_found = True
        for model_dir in required_models:
            model_path = cache_dir / model_dir
            if model_path.exists():
                snapshots_dir = model_path / "snapshots"
                if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                    print(f"   [OK] {model_dir[8:].replace('--', '/')}")
                else:
                    print(f"   [WARNING] {model_dir[8:].replace('--', '/')} (无快照)")
                    all_found = False
            else:
                print(f"   [ERROR] {model_dir[8:].replace('--', '/')} (未找到)")
                all_found = False
        
        if all_found:
            print("\n[SUCCESS] 所有必需模型都已找到！")
            print("\n下一步:")
            print("1. 运行: python use_local_models.py")
            print("2. 或运行: start_offline.bat")
            print("3. 开始使用本地缓存模式")
        else:
            print("\n[WARNING] 部分模型缺失")
            print("\n解决方案:")
            print("1. 下载缺失模型: python download_models.py")
            print("2. 或使用在线模式: set HF_HUB_OFFLINE=0 && python main.py")
    
    else:
        print("   [ERROR] 缓存目录不存在")
        print("\n解决方案:")
        print("1. 确保模型已下载到该目录")
        print("2. 或运行: python download_models.py")
        print("3. 或使用在线模式: set HF_HUB_OFFLINE=0 && python main.py")
    
    print("\n" + "=" * 60)
    print("环境变量设置:")
    print(f"HF_HOME: {cache_dir.parent}")
    print(f"TRANSFORMERS_CACHE: {cache_dir}")
    print(f"HF_HUB_OFFLINE: 1 (离线模式)")
    print("=" * 60)

if __name__ == "__main__":
    main()