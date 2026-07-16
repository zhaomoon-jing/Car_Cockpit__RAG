#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用本地缓存模型的统一入口点
运行此脚本将自动设置本地缓存环境并启动应用
"""

import os
import sys
from pathlib import Path

def setup_environment():
    """设置本地缓存环境"""
    
    # 默认缓存路径（根据您的系统调整）
    cache_path = Path("F:/ModelCache/huggingface/hub")
    
    print("=" * 60)
    print("🤖 汽车座舱RAG系统 - 本地缓存模式")
    print("=" * 60)
    
    # 检查缓存目录
    if not cache_path.exists():
        print(f"❌ 缓存目录不存在: {cache_path}")
        print()
        print("💡 解决方案:")
        print("1. 确保模型已下载到该目录")
        print("2. 或者指定正确的缓存路径")
        print("3. 运行以下命令下载模型:")
        print("   python download_models.py")
        print()
        
        # 创建目录
        try:
            cache_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 已创建缓存目录: {cache_path}")
            print("请下载模型到该目录后重新运行")
            return False
        except Exception as e:
            print(f"❌ 创建目录失败: {str(e)}")
            return False
    
    # 设置环境变量
    os.environ["HF_HOME"] = str(cache_path.parent)
    os.environ["TRANSFORMERS_CACHE"] = str(cache_path)
    os.environ["HF_DATASETS_CACHE"] = str(cache_path.parent / "datasets")
    os.environ["HF_MODULES_CACHE"] = str(cache_path.parent / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"  # 强制离线模式
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    print("✅ 环境变量已设置:")
    print(f"   HF_HOME: {os.environ['HF_HOME']}")
    print(f"   TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    print(f"   HF_HUB_OFFLINE: {os.environ['HF_HUB_OFFLINE']}")
    print()
    
    return True

def check_required_models():
    """检查必需模型是否在缓存中"""
    
    cache_dir = Path(os.environ.get("TRANSFORMERS_CACHE", ""))
    if not cache_dir.exists():
        print("❌ 缓存目录不存在")
        return False
    
    print("🔍 检查必需模型...")
    print("-" * 40)
    
    required_models = {
        "LLM模型 (Qwen/Qwen2.5-7B-Instruct)": "models--Qwen--Qwen2.5-7B-Instruct",
        "嵌入模型 (BAAI/bge-small-zh-v1.5)": "models--BAAI--bge-small-zh-v1.5",
        "语音识别模型 (openai/whisper-small)": "models--openai--whisper-small",
        "意图分类模型 (bert-base-chinese)": "models--bert-base-chinese"
    }
    
    all_available = True
    for model_name, cache_dir_name in required_models.items():
        model_path = cache_dir / cache_dir_name
        if model_path.exists():
            # 检查是否有快照
            snapshots_dir = model_path / "snapshots"
            if snapshots_dir.exists() and any(snapshots_dir.iterdir()):
                print(f"  ✅ {model_name}")
            else:
                print(f"  ⚠️  {model_name} (目录存在但无快照)")
                all_available = False
        else:
            print(f"  ❌ {model_name} (未找到)")
            all_available = False
    
    print()
    return all_available

def run_application():
    """运行应用程序"""
    
    print("🚀 启动汽车座舱RAG系统...")
    print("-" * 60)
    
    try:
        # 添加项目路径
        sys.path.append(str(Path(__file__).parent))
        
        # 导入并运行主程序
        from main import main
        main()
        
    except ImportError as e:
        print(f"❌ 导入错误: {str(e)}")
        print("请确保已安装所有依赖: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ 启动失败: {str(e)}")
        return False
    
    return True

def main():
    """主函数"""
    
    # 设置环境
    if not setup_environment():
        sys.exit(1)
    
    # 检查模型
    models_available = check_required_models()
    
    if not models_available:
        print("⚠️ 警告：部分必需模型未在本地缓存中找到")
        print()
        print("📥 下载缺失模型:")
        print("1. 临时启用在线模式:")
        print("   set HF_HUB_OFFLINE=0")
        print("   python main.py")
        print()
        print("2. 使用下载脚本:")
        print("   python download_models.py")
        print()
        
        choice = input("是否继续启动？（可能无法加载某些模型）(y/n): ").strip().lower()
        if choice != 'y':
            print("启动已取消")
            sys.exit(0)
    
    # 运行应用
    print("=" * 60)
    success = run_application()
    
    if not success:
        print("❌ 应用程序启动失败")
        print()
        print("🔧 故障排除:")
        print("1. 检查Python版本: python --version (需要Python 3.8+)")
        print("2. 安装依赖: pip install -r requirements.txt")
        print("3. 下载模型: python download_models.py")
        print("4. 使用在线模式: set HF_HUB_OFFLINE=0")
        print("5. 使用简化版本: python main_simple.py")
        sys.exit(1)

if __name__ == "__main__":
    main()