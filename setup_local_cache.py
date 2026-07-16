#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置本地HuggingFace缓存环境
"""

import os
import sys
from pathlib import Path

def setup_local_cache_env():
    """设置本地缓存环境变量"""
    
    # HuggingFace缓存路径 - 指向F:\ModelCache\huggingface\hub
    default_cache_path = Path("F:/ModelCache/huggingface/hub")
    
    # 检查缓存目录是否存在
    if not default_cache_path.exists():
        print(f"❌ HuggingFace缓存目录不存在: {default_cache_path}")
        print("请先下载模型到该目录或指定正确的路径")
        
        # 询问用户是否创建目录
        choice = input("是否创建目录? (y/n): ").strip().lower()
        if choice == 'y':
            default_cache_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ 已创建目录: {default_cache_path}")
        else:
            print("请手动下载模型或指定正确的缓存路径")
            return False
    
    # 设置环境变量
    os.environ["HF_HOME"] = str(default_cache_path.parent)
    os.environ["TRANSFORMERS_CACHE"] = str(default_cache_path)
    os.environ["HF_DATASETS_CACHE"] = str(default_cache_path.parent / "datasets")
    os.environ["HF_MODULES_CACHE"] = str(default_cache_path.parent / "modules")
    os.environ["HF_HUB_OFFLINE"] = "1"  # 离线模式
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    
    print("=" * 60)
    print("✅ 本地缓存环境已设置")
    print("=" * 60)
    print(f"📁 缓存目录: {default_cache_path}")
    print(f"💾 离线模式: 已启用 (HF_HUB_OFFLINE=1)")
    print()
    print("📋 已设置的环境变量:")
    print(f"  HF_HOME: {os.environ['HF_HOME']}")
    print(f"  TRANSFORMERS_CACHE: {os.environ['TRANSFORMERS_CACHE']}")
    print(f"  HF_DATASETS_CACHE: {os.environ['HF_DATASETS_CACHE']}")
    print(f"  HF_MODULES_CACHE: {os.environ['HF_MODULES_CACHE']}")
    print(f"  HF_HUB_OFFLINE: {os.environ['HF_HUB_OFFLINE']}")
    print()
    
    return True

def check_local_models():
    """检查本地缓存中的模型"""
    
    cache_dir = Path(os.environ.get("TRANSFORMERS_CACHE", ""))
    if not cache_dir.exists():
        print("❌ TRANSFORMERS_CACHE目录不存在")
        return
    
    print("🔍 检查本地缓存中的模型...")
    print("-" * 40)
    
    # 查找模型目录
    model_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith("models--")]
    
    if not model_dirs:
        print("❌ 未找到任何模型缓存")
        return
    
    print(f"📊 找到 {len(model_dirs)} 个模型缓存:")
    print()
    
    for model_dir in model_dirs:
        model_name = model_dir.name[8:].replace("--", "/")  # 移除 "models--" 前缀
        
        # 检查快照
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = list(snapshots_dir.iterdir())
            if snapshots:
                # 获取最新快照
                latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
                snapshot_size = sum(f.stat().st_size for f in latest_snapshot.rglob("*") if f.is_file())
                
                print(f"  ✅ {model_name}")
                print(f"     快照: {len(snapshots)} 个")
                print(f"     最新: {latest_snapshot.name}")
                print(f"     大小: {snapshot_size / (1024*1024):.1f} MB")
                print(f"     路径: {latest_snapshot}")
                print()
            else:
                print(f"  ⚠️  {model_name} (无快照)")
                print()
        else:
            print(f"  ⚠️  {model_name} (快照目录不存在)")
            print()

def main():
    """主函数"""
    print("=" * 60)
    print("🤖 本地HuggingFace缓存设置工具")
    print("=" * 60)
    print()
    
    # 设置环境
    if not setup_local_cache_env():
        sys.exit(1)
    
    # 检查模型
    check_local_models()
    
    print("=" * 60)
    print("🎯 使用方法:")
    print("=" * 60)
    print()
    print("1. 启动本地缓存模式:")
    print("   直接运行: python main.py")
    print("   或运行: python setup_local_cache.py")
    print()
    print("2. 临时启用在线模式（下载新模型）:")
    print("   set HF_HUB_OFFLINE=0")
    print("   python main.py")
    print()
    print("3. 使用批处理文件:")
    print("   start_local.bat (Windows)")
    print()
    print("4. 管理缓存:")
    print("   python local_model_cache.py")
    print()
    print("💡 提示:")
    print("   - 确保模型已下载到缓存目录")
    print("   - 离线模式下，模型加载速度更快")
    print("   - 需要更新模型时，启用在线模式")
    print()
    
    # 询问是否创建启动脚本
    choice = input("是否创建启动脚本? (y/n): ").strip().lower()
    if choice == 'y':
        create_startup_scripts()
        print("✅ 启动脚本已创建")

def create_startup_scripts():
    """创建启动脚本"""
    
    cache_dir = Path(os.environ.get("TRANSFORMERS_CACHE", ""))
    
    # Windows批处理文件
    bat_content = f"""@echo off
chcp 65001 >nul

echo ========================================
echo 汽车座舱RAG系统 - 本地缓存模式启动脚本
echo ========================================
echo.

echo [1/3] 设置本地缓存环境...
set HF_HOME={cache_dir.parent}
set TRANSFORMERS_CACHE={cache_dir}
set HF_DATASETS_CACHE={cache_dir.parent}\\datasets
set HF_MODULES_CACHE={cache_dir.parent}\\modules
set HF_HUB_OFFLINE=1
echo ✅ 环境变量设置完成

echo.
echo [2/3] 检查Python环境...
python --version
if errorlevel 1 (
    echo ❌ 未找到Python，请安装Python 3.8+
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
    
    bat_path = Path(__file__).parent / "start_local_cache.bat"
    with open(bat_path, "w", encoding="utf-8") as f:
        f.write(bat_content)
    
    print(f"✅ Windows批处理文件已创建: {bat_path}")
    
    # Python脚本
    py_content = f"""#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汽车座舱RAG系统 - 本地缓存启动脚本
"""

import os
import sys
from pathlib import Path

# 设置本地缓存环境
os.environ["HF_HOME"] = r"{cache_dir.parent}"
os.environ["TRANSFORMERS_CACHE"] = r"{cache_dir}"
os.environ["HF_DATASETS_CACHE"] = r"{cache_dir.parent}\\datasets"
os.environ["HF_MODULES_CACHE"] = r"{cache_dir.parent}\\modules"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

print("=" * 60)
print("🚀 汽车座舱RAG系统 - 本地缓存模式")
print("=" * 60)
print(f"📁 缓存目录: {cache_dir}")
print("💾 模式: 离线（仅使用本地缓存）")
print()

# 导入并运行主程序
try:
    from main import main
    main()
except KeyboardInterrupt:
    print("\\n👋 程序已停止")
except Exception as e:
    print(f"❌ 启动失败: {{str(e)}}")
"""
    
    py_path = Path(__file__).parent / "start_local_cache.py"
    with open(py_path, "w", encoding="utf-8") as f:
        f.write(py_content)
    
    print(f"✅ Python启动脚本已创建: {py_path}")

if __name__ == "__main__":
    main()