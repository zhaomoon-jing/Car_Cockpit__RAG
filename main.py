#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汽车座舱RAG系统 - 一键启动Gradio应用
"""

import sys
import os
from pathlib import Path

# 禁止字节码缓存，确保加载最新代码
sys.dont_write_bytecode = True

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

# 自动检查并修复 app.py 中的已知问题
def _check_and_fix_app():
    app_path = Path(__file__).parent / 'gradio_web' / 'app.py'
    if not app_path.exists():
        return
    with open(app_path, 'r', encoding='utf-8') as f:
        content = f.read()
    fixed = False
    # 移除不兼容的 type="messages" 参数
    if 'type="messages"' in content or "type='messages'" in content:
        content = content.replace('type="messages"', '')
        content = content.replace("type='messages'", '')
        fixed = True
    if fixed:
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("[FIX] 已自动修复 app.py 中的兼容性问题")
    # 清除 __pycache__
    import shutil
    for root, dirs, files in os.walk(Path(__file__).parent):
        for d in dirs:
            if d == '__pycache__':
                try:
                    shutil.rmtree(os.path.join(root, d))
                except Exception:
                    pass

_check_and_fix_app()

from gradio_web.app import create_app
import config

def main():
    """
    主函数：一键启动Gradio应用
    """
    print("[CAR] 启动汽车座舱RAG系统...")
    print("=" * 50)
    
    # 检查必要的目录和文件
    print("[DIR] 检查项目结构...")
    required_dirs = [
        config.DATA_DIR,
        config.RAW_DATA_DIR,
        config.CHUNKS_DIR,
        config.VECTOR_STORE_DIR,
        config.FAISS_INDEX_DIR,
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists():
            print(f"  [OK] {dir_path.relative_to(config.ROOT_DIR)}")
        else:
            print(f"  [WARN] {dir_path.relative_to(config.ROOT_DIR)} - 创建中...")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # 检查配置文件
    print("\n[CFG] 检查配置...")
    print(f"  嵌入模型: {config.MODEL_CONFIG['embedding_model']}")
    print(f"  LLM模型: {config.MODEL_CONFIG['llm_model']}")
    print(f"  分块大小: {config.RAG_CONFIG['chunk_size']}")
    print(f"  检索数量: {config.RAG_CONFIG['top_k']}")
    
    # 检查向量索引是否存在
    faiss_index_files = list(config.FAISS_INDEX_DIR.glob("*"))
    # 过滤出有效的FAISS索引文件
    valid_index_files = []
    for file in faiss_index_files:
        if file.is_file():
            if file.name.endswith(".bin") or file.name.endswith(".index"):
                valid_index_files.append(file)
    
    if valid_index_files:
        print(f"\n[INDEX] 找到向量索引文件: {len(valid_index_files)}个")
        for file in valid_index_files[:3]:  # 显示前3个
            print(f"  - {file.name}")
        if len(valid_index_files) > 3:
            print(f"  ... 和 {len(valid_index_files) - 3} 个其他文件")
        
        # 检查是否有完整的索引文件集合
        index_bin = config.FAISS_INDEX_DIR / "faiss_index.bin"
        metadata_pkl = config.FAISS_INDEX_DIR / "metadata.pkl"
        config_json = config.FAISS_INDEX_DIR / "config.json"
        
        if index_bin.exists() and metadata_pkl.exists():
            print(f"\n[OK] 检测到完整的FAISS索引:")
            print(f"   - {index_bin.name} - FAISS向量索引")
            print(f"   - {metadata_pkl.name} - 元数据文件")
            if config_json.exists():
                print(f"   - {config_json.name} - 配置文件")
            print(f"  索引已就绪，可以正常使用RAG功能")
        else:
            print(f"\n[WARN] 警告: 索引文件不完整")
            if not index_bin.exists():
                print(f"   [ERROR] 缺少: {index_bin.name}")
            if not metadata_pkl.exists():
                print(f"   [ERROR] 缺少: {metadata_pkl.name}")
            print("\n  请运行 python vector_store/build_faiss.py 重新构建索引")
    else:
        print("\n[WARN] 警告: 未找到向量索引文件")
        print("  请先运行数据预处理和向量化流程:")
        print("  1. 将车辆手册PDF/TXT文件放入 data/raw/ 目录")
        print("  2. 运行 python data_process/parse_pdf.py")
        print("  3. 运行 python data_process/build_chunk.py")
        print("  4. 运行 python vector_store/build_faiss.py")
        print("\n  或者直接使用基础功能（无向量检索）")
    
    # 创建并启动Gradio应用
    print("\n[WEB] 启动Gradio Web界面...")
    print("=" * 50)
    
    app = create_app()
    
    # 获取本地访问地址
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print(f"\n[OK] 应用已启动!")
    print(f"  本地访问: http://localhost:7860")
    print(f"  网络访问: http://{local_ip}:7860")
    print("\n[INFO] 使用说明:")
    print("  1. 在左侧上传车辆手册文件")
    print("  2. 点击'处理文档'按钮进行向量化")
    print("  3. 在右侧输入问题与车辆手册对话")
    print("  4. 支持语音输入和意图识别")
    print("\n按 Ctrl+C 停止服务器")
    
    # 启动应用
    print(f"\n[INFO] 正在启动Gradio服务器...")
    print(f"  本地访问: http://localhost:7860")
    print(f"  网络访问: http://{local_ip}:7860")
    print("\n按 Ctrl+C 停止服务器")
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
        quiet=False  # 显示启动信息
    )

if __name__ == "__main__":
    main()