#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动下载模型脚本 - 解决网络连接问题
"""

import os
import sys
import time
import requests
from pathlib import Path

# 设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5分钟超时

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 模型列表（按大小排序，小模型先下载）
MODELS = [
    # 小模型（快速测试）
    {
        "name": "bert-base-chinese",
        "type": "text-classification",
        "description": "中文意图分类模型 (415MB)",
        "priority": 1
    },
    {
        "name": "BAAI/bge-small-zh-v1.5",
        "type": "embedding",
        "description": "中文文本嵌入模型 (133MB)",
        "priority": 2
    },
    {
        "name": "openai/whisper-small",
        "type": "speech-recognition",
        "description": "语音识别模型 (1.1GB)",
        "priority": 3
    },
    # 中等模型
    {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "type": "text-generation",
        "description": "通义千问小模型 (3.2GB)",
        "priority": 4
    },
    {
        "name": "Qwen/Qwen2.5-0.5B-Instruct",
        "type": "text-generation",
        "description": "通义千问小模型",
        "priority": 5
    },
    # 大模型（如果小模型不够用）
    {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "type": "text-generation",
        "description": "通义千问完整模型 (14GB)",
        "priority": 6,
        "optional": True  # 可选，如果小模型够用可以不下载
    },
]

def test_connection(url, timeout=10):
    """测试连接"""
    try:
        response = requests.head(url, timeout=timeout)
        return response.status_code < 400
    except:
        return False

def download_model(model_name, model_type, cache_dir):
    """下载单个模型"""
    print(f"\n{'='*60}")
    print(f"📥 下载模型: {model_name}")
    print(f"📝 描述: {model_type}")
    print(f"📁 缓存目录: {cache_dir}")
    print(f"{'='*60}")
    
    # 创建缓存目录
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if model_type == "text-generation":
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print("1. 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ Tokenizer下载完成")
            
            print("2. 下载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ 模型下载完成")
            
        elif model_type == "text-classification":
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print("1. 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ Tokenizer下载完成")
            
            print("2. 下载模型...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ 模型下载完成")
            
        elif model_type == "embedding":
            from sentence_transformers import SentenceTransformer
            
            print("下载SentenceTransformer模型...")
            model = SentenceTransformer(
                model_name,
                cache_folder=str(cache_dir)
            )
            print("   ✅ 模型下载完成")
            
        elif model_type == "speech-recognition":
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
            
            print("1. 下载processor...")
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ Processor下载完成")
            
            print("2. 下载模型...")
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ 模型下载完成")
            
        else:
            from transformers import AutoTokenizer, AutoModel
            
            print("1. 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ Tokenizer下载完成")
            
            print("2. 下载模型...")
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                resume_download=True,
                local_files_only=False
            )
            print("   ✅ 模型下载完成")
        
        print(f"✅ {model_name} 下载成功！")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {str(e)[:200]}")
        return False

def main():
    """主函数"""
    print("🚀 汽车座舱RAG系统 - 模型下载工具")
    print("="*60)
    
    # 测试网络连接
    print("🔍 测试网络连接...")
    mirrors = [
        ("HF Mirror", "https://hf-mirror.com"),
        ("上海交大镜像", "https://mirror.sjtu.edu.cn"),
        ("HuggingFace官方", "https://huggingface.co"),
    ]
    
    available_mirrors = []
    for name, url in mirrors:
        if test_connection(url):
            print(f"   ✅ {name}: 可用")
            available_mirrors.append(url)
        else:
            print(f"   ❌ {name}: 不可用")
    
    if not available_mirrors:
        print("\n⚠️  所有镜像源都不可用，请检查网络连接！")
        print("   可以尝试：")
        print("   1. 使用VPN或代理")
        print("   2. 设置HTTP代理：set HTTP_PROXY=http://127.0.0.1:7890")
        print("   3. 使用离线模式（需要预先下载模型）")
        return
    
    # 选择最快的镜像
    fastest_mirror = available_mirrors[0]
    print(f"\n🌐 使用镜像源: {fastest_mirror}")
    os.environ["HF_ENDPOINT"] = fastest_mirror
    
    # 选择下载目录
    cache_dir = ROOT_DIR / "models_cache"
    print(f"📁 缓存目录: {cache_dir}")
    
    # 询问用户选择
    print("\n📋 可用模型:")
    for i, model in enumerate(MODELS, 1):
        optional = " (可选)" if model.get("optional", False) else ""
        print(f"  {i}. {model['name']}")
        print(f"     类型: {model['type']} | {model['description']}{optional}")
    
    print("\n🔧 下载选项:")
    print("  1. 下载所有必需模型（推荐）")
    print("  2. 下载所有模型（包括可选大模型）")
    print("  3. 自定义选择模型")
    print("  4. 仅测试连接，不下载")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    models_to_download = []
    
    if choice == "1":
        # 下载必需模型（排除可选的大模型）
        models_to_download = [m for m in MODELS if not m.get("optional", False)]
    elif choice == "2":
        # 下载所有模型
        models_to_download = MODELS.copy()
    elif choice == "3":
        # 自定义选择
        print("\n选择要下载的模型（输入数字，用逗号分隔，如：1,2,3）:")
        selections = input("选择: ").strip().split(",")
        for sel in selections:
            try:
                idx = int(sel.strip()) - 1
                if 0 <= idx < len(MODELS):
                    models_to_download.append(MODELS[idx])
            except:
                pass
    elif choice == "4":
        print("✅ 连接测试完成")
        return
    else:
        print("❌ 无效选择，使用默认选项1")
        models_to_download = [m for m in MODELS if not m.get("optional", False)]
    
    if not models_to_download:
        print("❌ 没有选择任何模型")
        return
    
    # 按优先级排序
    models_to_download.sort(key=lambda x: x["priority"])
    
    print(f"\n📥 准备下载 {len(models_to_download)} 个模型...")
    print("="*60)
    
    success_count = 0
    failed_models = []
    
    for model in models_to_download:
        success = download_model(model["name"], model["type"], cache_dir)
        if success:
            success_count += 1
        else:
            failed_models.append(model["name"])
        
        # 等待一下，避免请求过于频繁
        time.sleep(1)
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 下载完成！")
    print(f"✅ 成功: {success_count}/{len(models_to_download)}")
    
    if failed_models:
        print(f"❌ 失败: {len(failed_models)}")
        for model_name in failed_models:
            print(f"   - {model_name}")
        
        print("\n💡 解决方案:")
        print("   1. 检查网络连接")
        print("   2. 设置代理: set HTTP_PROXY=http://127.0.0.1:7890")
        print("   3. 手动下载模型到: models_cache/ 目录")
        print("   4. 使用更小的模型版本")
    
    print(f"\n📁 模型缓存位置: {cache_dir}")
    print("🔧 使用说明:")
    print("   1. 设置环境变量: set TRANSFORMERS_CACHE=models_cache")
    print("   2. 运行程序: python main.py")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断下载")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n💡 请确保已安装依赖:")
        print("   pip install transformers sentence-transformers torch")
        print("   或运行: pip install -r requirements.txt")