#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线模式配置 - 当网络连接失败时使用
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 本地模型路径配置
LOCAL_MODEL_PATHS = {
    # 文本生成模型
    "Qwen/Qwen2.5-7B-Instruct": {
        "local_path": Path("F:/ModelCache/huggingface/hub") / "models--Qwen--Qwen2.5-7B-Instruct",
        "backup": "Qwen/Qwen2.5-1.5B-Instruct",
        "backup_local": Path("F:/ModelCache/huggingface/hub") / "models--Qwen--Qwen2.5-1.5B-Instruct"
    },
    
    # 嵌入模型
    "BAAI/bge-small-zh-v1.5": {
        "local_path": Path("F:/ModelCache/huggingface/hub") / "models--BAAI--bge-small-zh-v1.5",
        "backup": "sentence-transformers/all-MiniLM-L6-v2",
        "backup_local": Path("F:/ModelCache/huggingface/hub") / "models--sentence-transformers--all-MiniLM-L6-v2"
    },
    
    # 意图分类模型
    "bert-base-chinese": {
        "local_path": Path("F:/ModelCache/huggingface/hub") / "models--bert-base-chinese",
        "backup": "hfl/chinese-roberta-wwm-ext",
        "backup_local": Path("F:/ModelCache/huggingface/hub") / "models--hfl--chinese-roberta-wwm-ext"
    },
    
    # 语音识别模型
    "openai/whisper-small": {
        "local_path": Path("F:/ModelCache/huggingface/hub") / "models--openai--whisper-small",
        "backup": "openai/whisper-tiny",
        "backup_local": Path("F:/ModelCache/huggingface/hub") / "models--openai--whisper-tiny"
    }
}

# 模型下载状态
MODEL_DOWNLOAD_STATUS = {
    "Qwen/Qwen2.5-7B-Instruct": False,
    "BAAI/bge-small-zh-v1.5": False,
    "bert-base-chinese": False,
    "openai/whisper-small": False,
}

def check_local_models():
    """检查本地模型是否存在"""
    print("🔍 检查本地模型...")
    
    for model_name, config in LOCAL_MODEL_PATHS.items():
        local_path = config["local_path"]
        backup_local = config.get("backup_local")
        
        if local_path.exists():
            print(f"   ✅ {model_name}: 本地模型存在 ({local_path})")
            MODEL_DOWNLOAD_STATUS[model_name] = True
        elif backup_local and backup_local.exists():
            print(f"   ⚠️  {model_name}: 主模型不存在，但备用模型存在")
            print(f"       备用: {config['backup']} ({backup_local})")
            MODEL_DOWNLOAD_STATUS[model_name] = True
        else:
            print(f"   ❌ {model_name}: 本地模型不存在")
            print(f"       路径: {local_path}")
            MODEL_DOWNLOAD_STATUS[model_name] = False
    
    return MODEL_DOWNLOAD_STATUS

def get_local_model_path(model_name):
    """获取本地模型路径"""
    config = LOCAL_MODEL_PATHS.get(model_name)
    if not config:
        return None
    
    # 检查主模型路径
    local_path = config["local_path"]
    if local_path.exists():
        return str(local_path)
    
    # 检查备用模型路径
    backup_local = config.get("backup_local")
    if backup_local and backup_local.exists():
        return str(backup_local)
    
    return None

def setup_offline_mode():
    """设置离线模式"""
    print("🔄 设置离线模式...")
    
    # 禁用网络下载
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_EVALUATE_OFFLINE"] = "1"
    
    print("   ✅ TRANSFORMERS_OFFLINE=1")
    print("   ✅ HF_DATASETS_OFFLINE=1")
    print("   ✅ HF_EVALUATE_OFFLINE=1")
    
    # 设置缓存路径
    cache_dir = Path("F:/ModelCache/huggingface/hub")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    os.environ["TRANSFORMERS_CACHE"] = str(cache_dir)
    os.environ["HF_HOME"] = str(cache_dir.parent)
    
    print(f"   ✅ TRANSFORMERS_CACHE={cache_dir}")
    print(f"   ✅ HF_HOME={cache_dir}")
    
    return cache_dir

def create_model_symlinks():
    """创建模型符号链接（如果需要）"""
    print("🔗 创建模型符号链接...")
    
    cache_dir = Path("F:/ModelCache/huggingface/hub")
    
    for model_name, config in LOCAL_MODEL_PATHS.items():
        local_path = config["local_path"]
        backup_local = config.get("backup_local")
        
        # 如果主模型不存在但备用模型存在，创建符号链接
        if not local_path.exists() and backup_local and backup_local.exists():
            try:
                import os
                if hasattr(os, "symlink"):
                    os.symlink(backup_local, local_path)
                    print(f"   ✅ 创建符号链接: {local_path.name} -> {backup_local.name}")
                else:
                    # Windows可能不支持symlink，使用复制或重命名
                    import shutil
                    shutil.copytree(backup_local, local_path)
                    print(f"   ✅ 复制备用模型: {backup_local.name} -> {local_path.name}")
            except Exception as e:
                print(f"   ⚠️  无法创建符号链接: {str(e)}")

def print_offline_instructions():
    """打印离线使用说明"""
    print("\n" + "="*60)
    print("📚 离线模式使用说明")
    print("="*60)
    
    print("\n1. 📥 下载模型:")
    print("   运行: python download_models.py")
    print("   选择选项1或2下载必需模型")
    
    print("\n2. 🛠️ 设置离线模式:")
    print("   在代码中添加:")
    print("   ```python")
    print("   import offline_config")
    print("   offline_config.setup_offline_mode()")
    print("   ```")
    
    print("\n3. 🔧 修改配置:")
    print("   编辑 config.py，将模型路径改为本地路径:")
    print("   ```python")
    print("   MODEL_CONFIG = {")
    print('       "llm_model": "F:/ModelCache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct",')
    print('       "embedding_model": "F:/ModelCache/huggingface/hub/models--BAAI--bge-small-zh-v1.5",')
    print('       "intent_model": "F:/ModelCache/huggingface/hub/models--bert-base-chinese",')
    print('       "asr_model": "F:/ModelCache/huggingface/hub/models--openai--whisper-small",')
    print("   }")
    print("   ```")
    
    print("\n4. 🚀 启动应用:")
    print("   设置环境变量:")
    print("   ```bash")
    print("   set TRANSFORMERS_OFFLINE=1")
    print("   set HF_DATASETS_OFFLINE=1")
    print("   python main.py")
    print("   ```")
    
    print("\n5. 💡 手动下载（如果自动下载失败）:")
    print("   a. 访问 https://huggingface.co/ 搜索模型")
    print("   b. 点击 'Files and versions'")
    print("   c. 下载所有文件到 models_cache/模型名/")
    print("   d. 确保目录结构正确")
    
    print("\n" + "="*60)

def main():
    """主函数"""
    print("🔧 离线模式配置工具")
    print("="*60)
    
    # 检查本地模型
    status = check_local_models()
    
    print(f"\n📊 模型状态:")
    available = sum(1 for v in status.values() if v)
    total = len(status)
    print(f"   可用: {available}/{total}")
    
    if available == total:
        print("✅ 所有模型都已就绪，可以启用离线模式")
        
        # 设置离线模式
        cache_dir = setup_offline_mode()
        
        # 创建符号链接
        create_model_symlinks()
        
        print("\n🎉 离线模式已就绪！")
        print(f"   模型目录: {cache_dir}")
        
    elif available > 0:
        print(f"⚠️  部分模型可用 ({available}/{total})")
        print("   可以启动应用，但部分功能可能受限")
        
        # 显示缺失的模型
        print("\n❌ 缺失的模型:")
        for model_name, is_available in status.items():
            if not is_available:
                print(f"   - {model_name}")
        
        cache_dir = setup_offline_mode()
        
        print("\n💡 建议:")
        print("   1. 运行 python download_models.py 下载缺失模型")
        print("   2. 或手动下载到: models_cache/ 目录")
        print(f"   3. 当前缓存目录: {cache_dir}")
        
    else:
        print("❌ 没有可用的本地模型")
        print("\n📥 需要下载模型:")
        for model_name in status.keys():
            print(f"   - {model_name}")
        
        print("\n🚀 运行以下命令下载模型:")
        print("   python download_models.py")
    
    # 打印使用说明
    print_offline_instructions()

if __name__ == "__main__":
    main()