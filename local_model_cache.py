#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地模型缓存工具 - 统一管理HuggingFace本地缓存
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入项目配置
sys.path.append(str(Path(__file__).parent))
try:
    from config import MODEL_CONFIG
except ImportError:
    # 如果无法导入config，使用默认配置
    MODEL_CONFIG = {
        "model_download": {
            "use_local_cache": True,
            "local_cache_path": "F:/ModelCache/huggingface/hub",
        }
    }


class LocalModelCache:
    """本地模型缓存管理类"""
    
    def __init__(self, cache_dir: str = None):
        """
        初始化本地缓存管理器
        
        Args:
            cache_dir: HuggingFace缓存目录，默认从配置读取
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(MODEL_CONFIG.get("model_download", {}).get("local_cache_path", 
                                "F:/ModelCache/huggingface/hub"))
        
        self.setup_cache_environment()
        
    def setup_cache_environment(self):
        """设置缓存环境变量"""
        if not self.cache_dir.exists():
            logger.warning(f"缓存目录不存在: {self.cache_dir}")
            logger.info("将使用默认HuggingFace缓存目录")
            return False
        
        # 设置环境变量
        os.environ["HF_HOME"] = str(self.cache_dir.parent)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(self.cache_dir.parent / "datasets")
        os.environ["HF_MODULES_CACHE"] = str(self.cache_dir.parent / "modules")
        os.environ["HF_HUB_OFFLINE"] = "1"  # 离线模式
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        
        logger.info(f"[OK] 本地缓存环境已设置: {self.cache_dir}")
        return True
    
    def enable_online_mode(self):
        """启用在线模式（用于下载新模型）"""
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        logger.info("🌐 启用在线模式，可以下载新模型")
    
    def disable_online_mode(self):
        """禁用在线模式（使用本地缓存）"""
        os.environ["HF_HUB_OFFLINE"] = "1"
        logger.info("💾 启用离线模式，仅使用本地缓存")
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        获取模型在本地缓存中的路径
        
        Args:
            model_name: HuggingFace模型名称，如 "Qwen/Qwen2.5-7B-Instruct"
        
        Returns:
            模型本地路径，如果未找到则返回None
        """
        # 将模型名称转换为缓存目录格式
        # 例如: Qwen/Qwen2.5-7B-Instruct -> models--Qwen--Qwen2.5-7B-Instruct
        cache_dir_name = model_name.replace("/", "--")
        if not cache_dir_name.startswith("models--"):
            cache_dir_name = f"models--{cache_dir_name}"
        
        model_cache_dir = self.cache_dir / cache_dir_name
        if not model_cache_dir.exists():
            logger.debug(f"模型缓存目录不存在: {model_cache_dir}")
            return None
        
        snapshots_dir = model_cache_dir / "snapshots"
        if not snapshots_dir.exists():
            logger.debug(f"模型快照目录不存在: {snapshots_dir}")
            return None
        
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            logger.debug(f"模型快照目录为空: {snapshots_dir}")
            return None
        
        # 使用最新的快照
        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
        return str(latest_snapshot)
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """
        列出本地缓存中可用的所有模型
        
        Returns:
            字典，键为模型类型，值为模型名称列表
        """
        if not self.cache_dir.exists():
            logger.warning(f"缓存目录不存在: {self.cache_dir}")
            return {}
        
        models_by_type = {
            "llm": [],
            "embedding": [],
            "asr": [],
            "intent": [],
            "other": []
        }
        
        # 遍历缓存目录
        for item in self.cache_dir.iterdir():
            if item.is_dir() and item.name.startswith("models--"):
                # 解析模型名称
                model_name = item.name[8:].replace("--", "/")  # 移除 "models--" 前缀
                
                # 分类模型
                if "qwen" in model_name.lower() or "chat" in model_name.lower() or "llama" in model_name.lower():
                    models_by_type["llm"].append(model_name)
                elif "bge" in model_name.lower() or "embedding" in model_name.lower():
                    models_by_type["embedding"].append(model_name)
                elif "whisper" in model_name.lower():
                    models_by_type["asr"].append(model_name)
                elif "bert" in model_name.lower() or "roberta" in model_name.lower():
                    models_by_type["intent"].append(model_name)
                else:
                    models_by_type["other"].append(model_name)
        
        return models_by_type
    
    def check_model_status(self, model_name: str) -> Dict[str, any]:
        """
        检查模型状态
        
        Args:
            model_name: 模型名称
        
        Returns:
            包含模型状态信息的字典
        """
        local_path = self.get_model_path(model_name)
        
        status = {
            "model": model_name,
            "has_local_cache": local_path is not None,
            "local_path": local_path,
            "cache_size": 0,
            "file_count": 0
        }
        
        if local_path:
            local_dir = Path(local_path)
            if local_dir.exists():
                # 计算缓存大小和文件数量
                total_size = 0
                file_count = 0
                for file_path in local_dir.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
                        file_count += 1
                
                status["cache_size"] = total_size
                status["file_count"] = file_count
                status["cache_size_mb"] = total_size / (1024 * 1024)
        
        return status
    
    def create_model_symlink(self, model_name: str, target_dir: Path) -> bool:
        """
        为模型创建符号链接到指定目录
        
        Args:
            model_name: 模型名称
            target_dir: 目标目录
        
        Returns:
            是否成功创建
        """
        local_path = self.get_model_path(model_name)
        if not local_path:
            logger.error(f"模型 {model_name} 未在本地缓存中找到")
            return False
        
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建符号链接
        link_path = target_dir / model_name.replace("/", "_")
        try:
            if link_path.exists():
                if link_path.is_symlink():
                    link_path.unlink()
                else:
                    logger.warning(f"目标路径已存在但不是符号链接: {link_path}")
                    return False
            
            # 创建符号链接
            os.symlink(local_path, link_path, target_is_directory=True)
            logger.info(f"✅ 已创建符号链接: {link_path} -> {local_path}")
            return True
            
        except Exception as e:
            logger.error(f"创建符号链接失败: {str(e)}")
            return False
    
    def verify_model_integrity(self, model_name: str) -> bool:
        """
        验证模型完整性
        
        Args:
            model_name: 模型名称
        
        Returns:
            模型是否完整
        """
        local_path = self.get_model_path(model_name)
        if not local_path:
            return False
        
        local_dir = Path(local_path)
        required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]
        
        for file in required_files:
            if not (local_dir / file).exists():
                logger.warning(f"模型文件缺失: {file}")
                return False
        
        return True


# 全局缓存管理器实例
_cache_manager = None

def get_cache_manager() -> LocalModelCache:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = LocalModelCache()
    return _cache_manager


def load_model_with_cache(model_name: str, model_class, **kwargs):
    """
    使用本地缓存加载模型
    
    Args:
        model_name: 模型名称
        model_class: 模型类（如 AutoTokenizer, AutoModelForCausalLM, SentenceTransformer等）
        **kwargs: 传递给模型类的参数
    
    Returns:
        加载的模型实例
    """
    # 检测是否为 SentenceTransformer（5.x 不支持 from_pretrained）
    is_sentence_transformer = False
    try:
        from sentence_transformers import SentenceTransformer
        is_sentence_transformer = (model_class is SentenceTransformer) or \
            (hasattr(model_class, '__name__') and model_class.__name__ == 'SentenceTransformer')
    except ImportError:
        pass
    
    cache_manager = get_cache_manager()
    local_path = cache_manager.get_model_path(model_name)
    
    if local_path:
        logger.info(f"🔍 从本地缓存加载模型: {model_name}")
        logger.info(f"📁 缓存路径: {local_path}")
        
        try:
            if is_sentence_transformer:
                # SentenceTransformer 5.x 用构造函数直接加载
                kwargs.pop("local_files_only", None)
                model = model_class(local_path, **kwargs)
            else:
                # transformers 类使用 from_pretrained
                kwargs["local_files_only"] = True
                model = model_class.from_pretrained(local_path, **kwargs)
            logger.info(f"✅ 模型 {model_name} 从本地缓存加载成功")
            return model
        except Exception as e:
            logger.warning(f"❌ 从本地缓存加载失败: {str(e)}，尝试在线下载")
            cache_manager.enable_online_mode()
            kwargs.pop("local_files_only", None)
    else:
        logger.info(f"🔍 本地缓存中未找到模型 {model_name}，尝试在线下载")
        cache_manager.enable_online_mode()
    
    # 在线下载
    try:
        if is_sentence_transformer:
            kwargs.pop("local_files_only", None)
            model = model_class(model_name, **kwargs)
        else:
            model = model_class.from_pretrained(model_name, **kwargs)
        logger.info(f"✅ 模型 {model_name} 下载成功")
        return model
    except Exception as e:
        logger.error(f"❌ 模型 {model_name} 加载失败: {str(e)}")
        raise


def main():
    """主函数：显示缓存信息和工具菜单"""
    print("=" * 60)
    print("本地HuggingFace缓存管理器")
    print("=" * 60)
    
    cache_manager = get_cache_manager()
    
    # 显示缓存信息
    print(f"📁 缓存目录: {cache_manager.cache_dir}")
    print(f"📊 缓存状态: {'可用' if cache_manager.cache_dir.exists() else '不可用'}")
    
    if not cache_manager.cache_dir.exists():
        print("\n❌ 缓存目录不存在，请检查路径或先下载模型")
        return
    
    # 列出可用模型
    print("\n🔍 扫描本地缓存中的模型...")
    models_by_type = cache_manager.list_available_models()
    
    total_models = sum(len(models) for models in models_by_type.values())
    print(f"📊 发现 {total_models} 个模型:")
    
    for model_type, models in models_by_type.items():
        if models:
            print(f"\n  {model_type.upper()} 模型 ({len(models)}个):")
            for model in models:
                status = cache_manager.check_model_status(model)
                size_mb = status.get("cache_size_mb", 0)
                status_str = "✅ 完整" if cache_manager.verify_model_integrity(model) else "⚠️ 不完整"
                print(f"    • {model} ({size_mb:.1f} MB) - {status_str}")
    
    # 检查项目所需模型
    print("\n🔧 检查项目所需模型:")
    required_models = {
        "llm": "Qwen/Qwen2.5-7B-Instruct",
        "embedding": "BAAI/bge-small-zh-v1.5",
        "asr": "openai/whisper-small",
        "intent": "bert-base-chinese"
    }
    
    all_available = True
    for model_type, model_name in required_models.items():
        status = cache_manager.check_model_status(model_name)
        if status["has_local_cache"]:
            print(f"  ✅ {model_type}: {model_name} - 已缓存")
        else:
            print(f"  ❌ {model_type}: {model_name} - 未找到")
            all_available = False
    
    print("\n" + "=" * 60)
    print("🛠️  工具菜单:")
    print("=" * 60)
    print("1. 启用离线模式（仅使用本地缓存）")
    print("2. 启用在线模式（可下载新模型）")
    print("3. 验证所有模型完整性")
    print("4. 创建项目符号链接")
    print("5. 生成启动脚本")
    print("6. 退出")
    
    try:
        choice = input("\n请选择操作 (1-6): ").strip()
        
        if choice == "1":
            cache_manager.disable_online_mode()
            print("✅ 已启用离线模式，将仅使用本地缓存")
            
        elif choice == "2":
            cache_manager.enable_online_mode()
            print("✅ 已启用在线模式，可以下载新模型")
            
        elif choice == "3":
            print("\n🔍 验证模型完整性...")
            for model_type, model_name in required_models.items():
                if cache_manager.verify_model_integrity(model_name):
                    print(f"  ✅ {model_name}: 完整")
                else:
                    print(f"  ❌ {model_name}: 不完整或缺失")
        
        elif choice == "4":
            print("\n🔗 创建项目符号链接...")
            project_dir = Path(__file__).parent / "local_models"
            if cache_manager.create_model_symlink("Qwen/Qwen2.5-7B-Instruct", project_dir):
                print(f"✅ LLM模型符号链接创建成功: {project_dir}")
            if cache_manager.create_model_symlink("BAAI/bge-small-zh-v1.5", project_dir):
                print(f"✅ 嵌入模型符号链接创建成功: {project_dir}")
            
        elif choice == "5":
            print("\n📜 生成启动脚本...")
            generate_startup_scripts(cache_manager)
            
        elif choice == "6":
            print("👋 再见！")
            
    except KeyboardInterrupt:
        print("\n\n👋 操作已取消")


def generate_startup_scripts(cache_manager: LocalModelCache):
    """生成启动脚本"""
    
    # Windows批处理文件
    bat_content = f"""@echo off
chcp 65001 >nul

echo ========================================
echo 汽车座舱RAG系统 - 本地缓存模式启动脚本
echo ========================================
echo.

echo [1/3] 设置本地缓存环境...
set HF_HOME={cache_manager.cache_dir.parent}
set TRANSFORMERS_CACHE={cache_manager.cache_dir}
set HF_DATASETS_CACHE={cache_manager.cache_dir.parent}\\datasets
set HF_MODULES_CACHE={cache_manager.cache_dir.parent}\\modules
set HF_HUB_OFFLINE=1
echo ✅ 环境变量设置完成

echo.
echo [2/3] 检查本地模型缓存...
python -c "
import sys
sys.path.append('.')
from local_model_cache import get_cache_manager

cache_manager = get_cache_manager()
models = cache_manager.list_available_models()

print('📁 本地缓存中的模型:')
for model_type, model_list in models.items():
    if model_list:
        print(f'  {{model_type.upper()}} ({{len(model_list)}}个):')
        for model in model_list[:3]:  # 只显示前3个
            print(f'    • {{model}}')
        if len(model_list) > 3:
            print(f'    ... 和 {{len(model_list)-3}} 个其他模型')
"
if errorlevel 1 (
    echo ❌ 检查本地模型缓存失败
    pause
    exit /b 1
)

echo.
echo [3/3] 启动应用...
echo ⚠️ 注意：使用本地缓存模型，无需网络下载
echo.

python main.py

echo.
echo ========================================
echo 应用已关闭
echo ========================================
pause
"""

    bat_path = Path(__file__).parent / "start_local.bat"
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)
    print(f"✅ Windows批处理文件已生成: {bat_path}")
    
    # Python启动脚本
    py_content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汽车座舱RAG系统 - 本地缓存启动脚本
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

# 设置本地缓存环境
os.environ["HF_HOME"] = r"{cache_manager.cache_dir.parent}"
os.environ["TRANSFORMERS_CACHE"] = r"{cache_manager.cache_dir}"
os.environ["HF_DATASETS_CACHE"] = r"{cache_manager.cache_dir.parent}\\datasets"
os.environ["HF_MODULES_CACHE"] = r"{cache_manager.cache_dir.parent}\\modules"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

print("=" * 60)
print("🚀 汽车座舱RAG系统 - 本地缓存模式")
print("=" * 60)
print(f"📁 缓存目录: {cache_manager.cache_dir}")
print("💾 模式: 离线（仅使用本地缓存）")
print()

# 检查必要模型
from local_model_cache import get_cache_manager
cache_manager = get_cache_manager()

required_models = {{
    "LLM模型": "Qwen/Qwen2.5-7B-Instruct",
    "嵌入模型": "BAAI/bge-small-zh-v1.5",
    "语音识别模型": "openai/whisper-small",
    "意图分类模型": "bert-base-chinese"
}}

print("🔍 检查必要模型:")
all_available = True
for name, model_name in required_models.items():
    local_path = cache_manager.get_model_path(model_name)
    if local_path:
        print(f"  ✅ {{name}}: {{model_name}}")
    else:
        print(f"  ❌ {{name}}: {{model_name}} (未找到)")
        all_available = False

if not all_available:
    print("\\n⚠️ 警告：部分模型未在本地缓存中找到")
    print("   请先下载模型或使用在线模式")
    choice = input("是否继续启动？(y/n): ")
    if choice.lower() != 'y':
        print("启动已取消")
        sys.exit(1)

print("\\n🚀 启动主程序...")
print("-" * 60)

# 导入并运行主程序
try:
    from main import main
    main()
except KeyboardInterrupt:
    print("\\n👋 程序已停止")
except Exception as e:
    print(f"❌ 启动失败: {{str(e)}}")
'''

    py_path = Path(__file__).parent / "start_local.py"
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(py_content)
    print(f"✅ Python启动脚本已生成: {py_path}")


if __name__ == "__main__":
    main()