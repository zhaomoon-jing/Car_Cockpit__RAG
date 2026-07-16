#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试模型加载 - 验证是否可以正常下载和加载模型
"""

import os
import sys
import time
from pathlib import Path

# 设置镜像源
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# 项目根目录
ROOT_DIR = Path(__file__).parent

def test_model_download(model_name, model_type="auto"):
    """测试单个模型下载"""
    print(f"\n{'='*60}")
    print(f"测试下载: {model_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        if model_type == "text-generation" or "Qwen" in model_name or "Chat" in model_name:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print("1. 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
                resume_download=True
            )
            print(f"   ✅ Tokenizer下载成功 ({time.time() - start_time:.1f}s)")
            
            print("2. 下载模型...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                resume_download=True
            )
            elapsed = time.time() - start_time
            print(f"   ✅ 模型下载成功 ({elapsed:.1f}s)")
            
            # 测试推理
            print("3. 测试推理...")
            test_text = "你好，这是一个测试。"
            inputs = tokenizer(test_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"   ✅ 推理测试成功")
            print(f"      输入: {test_text}")
            print(f"      输出: {response[:50]}...")
            
            return True
            
        elif model_type == "embedding" or "bge" in model_name:
            from sentence_transformers import SentenceTransformer
            
            print("下载SentenceTransformer模型...")
            model = SentenceTransformer(model_name)
            elapsed = time.time() - start_time
            print(f"   ✅ 模型下载成功 ({elapsed:.1f}s)")
            
            # 测试嵌入
            print("2. 测试嵌入...")
            embeddings = model.encode(["这是一个测试句子。"])
            print(f"   ✅ 嵌入测试成功")
            print(f"      嵌入维度: {embeddings.shape}")
            
            return True
            
        elif model_type == "intent" or "bert" in model_name:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            print("1. 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"   ✅ Tokenizer下载成功 ({time.time() - start_time:.1f}s)")
            
            print("2. 下载模型...")
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            elapsed = time.time() - start_time
            print(f"   ✅ 模型下载成功 ({elapsed:.1f}s)")
            
            # 测试推理
            print("3. 测试推理...")
            test_text = "这是一个测试文本"
            inputs = tokenizer(test_text, return_tensors="pt")
            outputs = model(**inputs)
            print(f"   ✅ 推理测试成功")
            
            return True
            
        else:
            from transformers import AutoTokenizer, AutoModel
            
            print("1. 下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"   ✅ Tokenizer下载成功 ({time.time() - start_time:.1f}s)")
            
            print("2. 下载模型...")
            model = AutoModel.from_pretrained(model_name)
            elapsed = time.time() - start_time
            print(f"   ✅ 模型下载成功 ({elapsed:.1f}s)")
            
            return True
            
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ 下载失败 ({elapsed:.1f}s): {str(e)[:200]}")
        return False

def test_connection():
    """测试网络连接"""
    import requests
    
    print("🔍 测试网络连接...")
    
    test_urls = [
        ("HF Mirror", "https://hf-mirror.com"),
        ("HuggingFace", "https://huggingface.co"),
        ("Google", "https://www.google.com"),
    ]
    
    all_ok = True
    for name, url in test_urls:
        try:
            response = requests.get(url, timeout=10)
            print(f"   ✅ {name}: 可访问 (状态码: {response.status_code})")
        except Exception as e:
            print(f"   ❌ {name}: 不可访问 - {str(e)[:100]}")
            all_ok = False
    
    return all_ok

def main():
    """主函数"""
    print("🚀 汽车座舱RAG系统 - 模型加载测试")
    print("="*60)
    
    # 测试网络连接
    if not test_connection():
        print("\n⚠️  网络连接有问题，请检查：")
        print("   1. 网络连接是否正常")
        print("   2. 是否可以使用VPN或代理")
        print("   3. 设置代理: set HTTP_PROXY=http://127.0.0.1:7890")
        print("\n💡 如果网络确实无法访问，请使用离线模式：")
        print("   1. 运行: python offline_config.py")
        print("   2. 按照说明手动下载模型")
        return
    
    # 测试小模型（下载更快）
    test_models = [
        ("bert-base-chinese", "intent"),
        ("BAAI/bge-small-zh-v1.5", "embedding"),
        ("Qwen/Qwen2.5-1.5B-Instruct", "text-generation"),
        ("openai/whisper-tiny", "speech"),
    ]
    
    print("\n📥 开始测试模型下载...")
    print("💡 提示: 首次下载可能需要一些时间，请耐心等待")
    print("="*60)
    
    success_count = 0
    for model_name, model_type in test_models:
        success = test_model_download(model_name, model_type)
        if success:
            success_count += 1
    
    # 总结
    print(f"\n{'='*60}")
    print("📊 测试结果总结")
    print(f"{'='*60}")
    print(f"✅ 成功: {success_count}/{len(test_models)}")
    
    if success_count == len(test_models):
        print("\n🎉 所有模型下载测试成功！")
        print("   可以正常运行汽车座舱RAG系统")
        print("\n🚀 启动命令:")
        print("   python main.py")
        print("   或双击 start_simple.bat")
    elif success_count >= 2:
        print(f"\n⚠️  部分模型下载成功 ({success_count}/{len(test_models)})")
        print("   系统可以启动，但部分功能可能受限")
        print("\n💡 建议:")
        print("   1. 运行完整下载: python download_models.py")
        print("   2. 或使用离线模式: python offline_config.py")
    else:
        print(f"\n❌ 大部分模型下载失败 ({success_count}/{len(test_models)})")
        print("\n🔧 解决方案:")
        print("   1. 设置代理: set HTTP_PROXY=http://127.0.0.1:7890")
        print("   2. 使用离线模式: python offline_config.py")
        print("   3. 手动下载模型到 models_cache/ 目录")
    
    print(f"\n📁 模型缓存目录: {ROOT_DIR / 'models_cache'}")
    print("🔧 环境变量设置:")
    print("   set HF_ENDPOINT=https://hf-mirror.com")
    print("   set HF_HUB_DOWNLOAD_TIMEOUT=120")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试出错: {str(e)}")
        print("\n💡 请确保已安装依赖:")
        print("   pip install transformers sentence-transformers torch requests")