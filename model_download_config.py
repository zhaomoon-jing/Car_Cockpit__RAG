#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型下载配置 - 解决HuggingFace连接超时问题
"""

import os
import sys
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 模型下载配置
MODEL_DOWNLOAD_CONFIG = {
    # HuggingFace镜像源
    "hf_mirrors": [
        "https://hf-mirror.com",  # 国内镜像1
        "https://mirror.sjtu.edu.cn/huggingface",  # 国内镜像2
        "https://huggingface.co",  # 官方源
    ],
    
    # 超时设置（秒）
    "timeout_settings": {
        "connect": 30,  # 连接超时
        "read": 60,     # 读取超时
        "total": 120,   # 总超时
    },
    
    # 重试设置
    "retry_settings": {
        "max_retries": 3,  # 最大重试次数
        "backoff_factor": 1.5,  # 退避因子
    },
    
    # 代理设置（如果需要）
    "proxy": None,  # 例如: "http://127.0.0.1:7890"
    
    # 本地模型路径映射
    "local_model_paths": {
        "Qwen/Qwen2.5-7B-Instruct": None,  # 本地路径或None表示从网络下载
        "BAAI/bge-small-zh-v1.5": None,
        "bert-base-chinese": None,
        "openai/whisper-small": None,
    },
    
    # 备用模型（如果主模型下载失败）
    "backup_models": {
        "Qwen/Qwen2.5-7B-Instruct": [
            "Qwen/Qwen2.5-1.5B-Instruct",  # 更小的版本
            "Qwen/Qwen2-7B-Chat",
            "THUDM/chatglm3-6b",
        ],
        "BAAI/bge-small-zh-v1.5": [
            "BAAI/bge-small-zh",
            "sentence-transformers/all-MiniLM-L6-v2",
        ],
    },
    
    # 模型缓存目录
    "cache_dir": "F:/ModelCache/huggingface/hub",
}

# 环境变量设置
def setup_environment():
    """设置下载环境变量"""
    # 设置HuggingFace镜像
    os.environ["HF_ENDPOINT"] = MODEL_DOWNLOAD_CONFIG["hf_mirrors"][0]
    
    # 设置缓存目录
    os.environ["TRANSFORMERS_CACHE"] = MODEL_DOWNLOAD_CONFIG["cache_dir"]
    os.environ["HF_HOME"] = MODEL_DOWNLOAD_CONFIG["cache_dir"]
    
    # 设置代理（如果有）
    if MODEL_DOWNLOAD_CONFIG["proxy"]:
        os.environ["HTTP_PROXY"] = MODEL_DOWNLOAD_CONFIG["proxy"]
        os.environ["HTTPS_PROXY"] = MODEL_DOWNLOAD_CONFIG["proxy"]
        os.environ["ALL_PROXY"] = MODEL_DOWNLOAD_CONFIG["proxy"]
    
    # 创建缓存目录
    cache_dir = Path(MODEL_DOWNLOAD_CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 环境变量已设置:")
    print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT')}")
    print(f"   缓存目录: {cache_dir}")
    if MODEL_DOWNLOAD_CONFIG["proxy"]:
        print(f"   代理: {MODEL_DOWNLOAD_CONFIG['proxy']}")

# 模型下载工具函数
def download_with_retry(model_name, **kwargs):
    """带重试机制的模型下载"""
    import time
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    
    setup_environment()
    
    max_retries = MODEL_DOWNLOAD_CONFIG["retry_settings"]["max_retries"]
    backoff_factor = MODEL_DOWNLOAD_CONFIG["retry_settings"]["backoff_factor"]
    
    for attempt in range(max_retries):
        try:
            print(f"尝试 {attempt + 1}/{max_retries}: 下载模型 {model_name}")
            
            # 检查是否有本地路径
            local_path = MODEL_DOWNLOAD_CONFIG["local_model_paths"].get(model_name)
            if local_path and Path(local_path).exists():
                print(f"使用本地模型: {local_path}")
                return local_path
            
            # 尝试下载
            if "CausalLM" in str(kwargs.get('cls', '')):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                    trust_remote_code=True,
                    resume_download=True,
                    **kwargs
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                    trust_remote_code=True,
                    resume_download=True,
                    **kwargs
                )
            
            print(f"✅ 模型下载成功: {model_name}")
            return model
            
        except Exception as e:
            print(f"❌ 下载失败 (尝试 {attempt + 1}): {str(e)}")
            
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                print(f"等待 {wait_time:.1f} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"⚠️ 所有重试均失败，尝试使用备用模型...")
                backup_models = MODEL_DOWNLOAD_CONFIG["backup_models"].get(model_name, [])
                if backup_models:
                    return try_backup_models(backup_models, **kwargs)
                else:
                    raise

def try_backup_models(backup_models, **kwargs):
    """尝试备用模型"""
    for backup_model in backup_models:
        try:
            print(f"尝试备用模型: {backup_model}")
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
            
            if "CausalLM" in str(kwargs.get('cls', '')):
                model = AutoModelForCausalLM.from_pretrained(
                    backup_model,
                    cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                    trust_remote_code=True,
                    **kwargs
                )
            else:
                model = AutoModel.from_pretrained(
                    backup_model,
                    cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                    trust_remote_code=True,
                    **kwargs
                )
            
            print(f"✅ 备用模型下载成功: {backup_model}")
            return model
            
        except Exception as e:
            print(f"❌ 备用模型 {backup_model} 下载失败: {str(e)}")
            continue
    
    raise Exception("所有模型下载均失败")

# 快速下载脚本
def download_all_models():
    """下载所有需要的模型"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
    
    setup_environment()
    
    models_to_download = [
        ("Qwen/Qwen2.5-7B-Instruct", AutoModelForCausalLM),
        ("BAAI/bge-small-zh-v1.5", AutoModel),
        ("bert-base-chinese", AutoModel),
        ("openai/whisper-small", AutoModel),
    ]
    
    for model_name, model_class in models_to_download:
        try:
            print(f"\n{'='*50}")
            print(f"下载模型: {model_name}")
            print(f"{'='*50}")
            
            # 下载tokenizer
            print("下载tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                trust_remote_code=True,
            )
            print(f"✅ Tokenizer下载成功")
            
            # 下载模型
            print("下载模型...")
            if model_class == AutoModelForCausalLM:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto" if torch.cuda.is_available() else None,
                )
            else:
                model = AutoModel.from_pretrained(
                    model_name,
                    cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                    trust_remote_code=True,
                )
            
            print(f"✅ 模型下载成功")
            
        except Exception as e:
            print(f"❌ 下载失败: {str(e)}")
            print(f"尝试备用模型...")
            
            backup_models = MODEL_DOWNLOAD_CONFIG["backup_models"].get(model_name, [])
            if backup_models:
                try:
                    backup_model = backup_models[0]
                    print(f"尝试备用模型: {backup_model}")
                    
                    if model_class == AutoModelForCausalLM:
                        model = AutoModelForCausalLM.from_pretrained(
                            backup_model,
                            cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                            trust_remote_code=True,
                            torch_dtype=torch.float16,
                            device_map="auto" if torch.cuda.is_available() else None,
                        )
                    else:
                        model = AutoModel.from_pretrained(
                            backup_model,
                            cache_dir=MODEL_DOWNLOAD_CONFIG["cache_dir"],
                            trust_remote_code=True,
                        )
                    
                    print(f"✅ 备用模型下载成功")
                    
                except Exception as e2:
                    print(f"❌ 备用模型也失败: {str(e2)}")
            else:
                print(f"⚠️ 没有可用的备用模型")

    print(f"\n{'='*50}")
    print("所有模型下载完成！")
    print(f"缓存目录: {MODEL_DOWNLOAD_CONFIG['cache_dir']}")
    print(f"{'='*50}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='模型下载配置工具')
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # setup命令
    setup_parser = subparsers.add_parser('setup', help='设置环境变量')
    
    # download命令
    download_parser = subparsers.add_parser('download', help='下载所有模型')
    
    # test命令
    test_parser = subparsers.add_parser('test', help='测试连接')
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_environment()
        print("✅ 环境设置完成")
        
    elif args.command == 'download':
        download_all_models()
        
    elif args.command == 'test':
        import requests
        
        setup_environment()
        
        print("测试HuggingFace连接...")
        for mirror in MODEL_DOWNLOAD_CONFIG["hf_mirrors"]:
            try:
                response = requests.get(mirror, timeout=10)
                print(f"✅ {mirror} - 连接成功 (状态码: {response.status_code})")
            except Exception as e:
                print(f"❌ {mirror} - 连接失败: {str(e)}")
                
    else:
        parser.print_help()