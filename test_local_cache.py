#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试本地缓存功能
"""

import os
import sys
from pathlib import Path

def test_local_cache_setup():
    """测试本地缓存设置"""
    print("=" * 60)
    print("测试本地缓存设置")
    print("=" * 60)
    
    # 设置环境变量
    cache_dir = Path("F:/ModelCache/huggingface/hub")
    
    os.environ["HF_HOME"] = str(cache_dir.parent)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir.parent / "datasets")
    os.environ["HF_MODULES_CACHE"] = str(cache_dir.parent / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    print(f"缓存目录: {cache_dir}")
    print(f"目录存在: {cache_dir.exists()}")
    
    if cache_dir.exists():
        print("\n已找到的模型:")
        for item in cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                model_name = item.name[8:].replace("--", "/")
                snapshots = list((item / "snapshots").iterdir()) if (item / "snapshots").exists() else []
                print(f"  - {model_name}: {len(snapshots)} 个快照")
    
    return cache_dir.exists()

def test_model_loading():
    """测试模型加载"""
    print("\n" + "=" * 60)
    print("测试模型加载")
    print("=" * 60)
    
    try:
        # 测试导入本地缓存模块
        from local_model_cache import LocalModelCache
        cache_manager = LocalModelCache()
        print("✅ 本地缓存管理器初始化成功")
        
        # 测试获取模型路径
        llm_path = cache_manager.get_model_path("Qwen/Qwen2.5-7B-Instruct")
        if llm_path:
            print(f"✅ LLM模型路径: {llm_path}")
        else:
            print("❌ LLM模型未找到")
        
        embedding_path = cache_manager.get_model_path("BAAI/bge-small-zh-v1.5")
        if embedding_path:
            print(f"✅ 嵌入模型路径: {embedding_path}")
        else:
            print("❌ 嵌入模型未找到")
            
    except Exception as e:
        print(f"❌ 模型加载测试失败: {str(e)}")
        return False
    
    return True

def test_rag_llm_import():
    """测试RAG LLM导入"""
    print("\n" + "=" * 60)
    print("测试RAG LLM模块导入")
    print("=" * 60)
    
    try:
        # 添加项目路径
        sys.path.append(str(Path(__file__).parent))
        
        # 测试导入rag_llm模块
        from llm_infer.rag_llm import RAGLLM
        print("✅ RAGLLM类导入成功")
        
        # 测试本地缓存函数导入
        from llm_infer.rag_llm import get_local_model_path
        print("✅ get_local_model_path函数导入成功")
        
        # 测试配置
        from config import MODEL_CONFIG
        print("✅ 配置导入成功")
        print(f"使用本地缓存: {MODEL_CONFIG.get('model_download', {}).get('use_local_cache', False)}")
        
    except ImportError as e:
        print(f"❌ 导入失败: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        return False
    
    return True

def test_offline_mode():
    """测试离线模式"""
    print("\n" + "=" * 60)
    print("测试离线模式")
    print("=" * 60)
    
    # 检查环境变量
    offline_mode = os.environ.get("HF_HUB_OFFLINE", "0")
    print(f"HF_HUB_OFFLINE: {offline_mode}")
    
    if offline_mode == "1":
        print("✅ 离线模式已启用")
        
        # 检查缓存目录设置
        cache_dir = os.environ.get("TRANSFORMERS_CACHE", "")
        print(f"TRANSFORMERS_CACHE: {cache_dir}")
        
        if cache_dir and Path(cache_dir).exists():
            print("✅ 缓存目录设置正确")
            return True
        else:
            print("❌ 缓存目录未设置或不存在")
            return False
    else:
        print("⚠️ 离线模式未启用")
        return False

def main():
    """主测试函数"""
    print("开始测试本地缓存功能")
    print()
    
    tests = [
        ("本地缓存设置", test_local_cache_setup),
        ("模型加载", test_model_loading),
        ("RAG LLM导入", test_rag_llm_import),
        ("离线模式", test_offline_mode),
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        try:
            passed = test_func()
            if passed:
                print(f"通过 {test_name}")
            else:
                print(f"失败 {test_name}")
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name}: 异常 - {str(e)}")
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("所有测试通过！")
        print("\n📋 下一步:")
        print("1. 运行: python use_local_models.py")
        print("2. 或运行: start_offline.bat")
        print("3. 开始使用本地缓存模式")
    else:
        print("部分测试失败")
        print("\n🔧 故障排除:")
        print("1. 确保模型已下载到缓存目录")
        print("2. 检查环境变量设置")
        print("3. 运行: python setup_local_cache.py")
        print("4. 查看: 本地缓存使用指南.md")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()