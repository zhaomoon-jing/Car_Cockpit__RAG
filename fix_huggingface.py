#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速修复HuggingFace连接问题
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """打印标题"""
    print("\n" + "="*60)
    print(text)
    print("="*60)

def run_command(cmd, check=True):
    """运行命令"""
    print(f"运行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ 成功")
            if result.stdout.strip():
                print(f"输出: {result.stdout[:200]}")
            return True
        else:
            print(f"❌ 失败 (退出码: {result.returncode})")
            if result.stderr.strip():
                print(f"错误: {result.stderr[:200]}")
            return False
    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return False

def fix_huggingface_connection():
    """修复HuggingFace连接"""
    print_header("🔧 修复HuggingFace连接问题")
    
    # 1. 设置环境变量
    print("\n1. 设置环境变量...")
    env_vars = {
        "HF_ENDPOINT": "https://hf-mirror.com",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        "TRANSFORMERS_CACHE": str(Path(__file__).parent / "models_cache"),
        "HF_HOME": str(Path(__file__).parent / "models_cache"),
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    # 2. 测试连接
    print("\n2. 测试HuggingFace连接...")
    import requests
    
    mirrors = [
        ("HF Mirror", "https://hf-mirror.com"),
        ("上海交大镜像", "https://mirror.sjtu.edu.cn/huggingface"),
        ("HuggingFace官方", "https://huggingface.co"),
    ]
    
    available = []
    for name, url in mirrors:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code < 400:
                print(f"   ✅ {name}: 可访问")
                available.append(url)
            else:
                print(f"   ⚠️  {name}: 状态码 {response.status_code}")
        except Exception as e:
            print(f"   ❌ {name}: 连接失败 - {str(e)[:50]}")
    
    if not available:
        print("\n❌ 所有镜像源都不可用")
        return False
    
    print(f"\n✅ 找到 {len(available)} 个可用镜像源")
    return True

def install_with_mirror():
    """使用镜像源安装依赖"""
    print_header("📦 使用镜像源安装依赖")
    
    # 使用清华镜像源
    mirror_url = "https://pypi.tuna.tsinghua.edu.cn/simple"
    
    packages = [
        "transformers==4.36.0",
        "torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        "gradio==4.19.0",
        "sentence-transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
        "langchain",
        "faiss-cpu",
        "rank-bm25",
        "jieba",
        "pypdf",
        "python-docx",
        "openai-whisper",
        "soundfile",
        "pydub",
    ]
    
    success_count = 0
    for package in packages:
        cmd = f"pip install {package} --index-url {mirror_url}"
        if run_command(cmd, check=False):
            success_count += 1
        else:
            # 如果使用镜像源失败，尝试官方源
            print(f"尝试使用官方源安装: {package}")
            cmd = f"pip install {package}"
            if run_command(cmd, check=False):
                success_count += 1
    
    print(f"\n📊 安装结果: {success_count}/{len(packages)} 成功")
    return success_count == len(packages)

def download_small_models():
    """下载小模型（快速测试）"""
    print_header("📥 下载小模型（快速测试）")
    
    # 创建缓存目录
    cache_dir = Path(__file__).parent / "models_cache"
    cache_dir.mkdir(exist_ok=True)
    
    small_models = [
        "bert-base-chinese",  # 415MB
        "sentence-transformers/all-MiniLM-L6-v2",  # 80MB
        "microsoft/phi-2",  # 2.7GB (但下载较快)
    ]
    
    import time
    from transformers import AutoTokenizer, AutoModel
    
    success_count = 0
    for model_name in small_models:
        print(f"\n下载: {model_name}")
        start_time = time.time()
        
        try:
            # 下载tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
                resume_download=True,
                timeout=60
            )
            print(f"   ✅ Tokenizer下载成功")
            
            # 下载模型
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=str(cache_dir),
                trust_remote_code=True,
                resume_download=True,
                timeout=120
            )
            
            elapsed = time.time() - start_time
            print(f"   ✅ 模型下载成功 ({elapsed:.1f}秒)")
            success_count += 1
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"   ❌ 下载失败 ({elapsed:.1f}秒): {str(e)[:100]}")
    
    print(f"\n📊 下载结果: {success_count}/{len(small_models)} 成功")
    return success_count > 0

def create_offline_config():
    """创建离线配置文件"""
    print_header("📄 创建离线配置文件")
    
    config_content = '''# 离线模式配置
# 将此文件重命名为 .env 或添加到环境变量

# HuggingFace设置
HF_ENDPOINT=https://hf-mirror.com
HF_HUB_DISABLE_SYMLINKS_WARNING=1
HF_HUB_DOWNLOAD_TIMEOUT=300

# 离线模式
TRANSFORMERS_OFFLINE=0  # 设置为1启用完全离线模式
HF_DATASETS_OFFLINE=0
HF_EVALUATE_OFFLINE=0

# 缓存目录
TRANSFORMERS_CACHE=./models_cache
HF_HOME=./models_cache

# 代理设置（如果需要）
# HTTP_PROXY=http://127.0.0.1:7890
# HTTPS_PROXY=http://127.0.0.1:7890
# ALL_PROXY=http://127.0.0.1:7890

# 备用模型
USE_BACKUP_MODELS=1  # 如果主模型下载失败，使用备用小模型

# 模型配置
LLM_MODEL=microsoft/phi-2  # 小模型，下载快
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
INTENT_MODEL=bert-base-chinese
'''
    
    config_file = Path(__file__).parent / "huggingface.env"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print(f"✅ 配置文件已创建: {config_file}")
    print("\n💡 使用方法:")
    print("   1. 设置环境变量: set TRANSFORMERS_CACHE=./models_cache")
    print("   2. 运行: python main.py")
    print("   3. 或运行: start_improved.bat")
    
    return True

def main():
    """主函数"""
    print("🚀 HuggingFace连接问题修复工具")
    print("="*60)
    
    print("\n选择修复选项:")
    print("  1. 快速修复（设置环境变量和镜像源）")
    print("  2. 重新安装依赖（使用镜像源）")
    print("  3. 下载小模型测试")
    print("  4. 创建离线配置文件")
    print("  5. 全部执行")
    print("  0. 退出")
    
    choice = input("\n请输入选项 (0-5): ").strip()
    
    if choice == "1":
        fix_huggingface_connection()
    elif choice == "2":
        install_with_mirror()
    elif choice == "3":
        download_small_models()
    elif choice == "4":
        create_offline_config()
    elif choice == "5":
        print_header("执行全部修复步骤")
        fix_huggingface_connection()
        install_with_mirror()
        download_small_models()
        create_offline_config()
    elif choice == "0":
        print("退出")
        return
    else:
        print("无效选项")
        return
    
    print_header("修复完成")
    print("\n🎉 修复步骤已完成！")
    print("\n下一步操作:")
    print("  1. 运行测试: python test_model_load.py")
    print("  2. 启动应用: python main.py")
    print("  3. 或使用批处理: start_improved.bat")
    print("\n💡 如果仍有问题:")
    print("  - 检查网络连接")
    print("  - 设置代理: set HTTP_PROXY=http://127.0.0.1:7890")
    print("  - 使用离线模式: python offline_config.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {str(e)}")
        print("\n💡 请确保已安装Python和pip")