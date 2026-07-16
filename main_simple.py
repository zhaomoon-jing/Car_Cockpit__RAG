#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汽车座舱RAG系统 - 简化版本（使用小模型）
"""

import sys
import os
import time
from pathlib import Path

# 设置环境变量（解决HuggingFace连接问题）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent))

def check_dependencies():
    """检查依赖包"""
    print("🔍 检查依赖包...")
    
    dependencies = [
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("gradio", "gradio"),
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("rank-bm25", "rank_bm25"),
        ("jieba", "jieba"),
    ]
    
    missing = []
    for pkg_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name}")
        except ImportError:
            print(f"  ❌ {pkg_name}")
            missing.append(pkg_name)
    
    if missing:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing)}")
        print("   请运行: pip install " + " ".join(missing))
        return False
    
    print("✅ 所有依赖包已安装")
    return True

def load_small_models():
    """加载小模型（避免下载大模型）"""
    print("\n📥 加载轻量级模型...")
    
    # 使用小模型配置
    small_model_config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # 80MB
        "llm_model": "microsoft/phi-2",  # 2.7GB，但下载稳定
        "intent_model": "bert-base-chinese",  # 415MB
        "asr_model": "openai/whisper-tiny",  # 151MB
    }
    
    print("  使用以下轻量级模型:")
    for key, value in small_model_config.items():
        print(f"    {key}: {value}")
    
    return small_model_config

def create_simple_app():
    """创建简化版应用"""
    print("\n🚀 创建简化版应用...")
    
    try:
        import gradio as gr
        from gradio_web.app import create_app
        
        # 修改配置使用小模型
        import config
        small_models = load_small_models()
        
        # 临时修改配置
        original_config = config.MODEL_CONFIG.copy()
        config.MODEL_CONFIG.update(small_models)
        
        print("✅ 配置已更新为轻量级模型")
        
        # 创建应用
        app = create_app()
        
        # 恢复原始配置
        config.MODEL_CONFIG.update(original_config)
        
        return app
        
    except Exception as e:
        print(f"❌ 创建应用失败: {str(e)}")
        print("\n💡 尝试创建基础版应用...")
        return create_basic_app()

def create_basic_app():
    """创建基础版应用（无模型依赖）"""
    print("🔄 创建基础版应用...")
    
    import gradio as gr
    
    with gr.Blocks(title="汽车座舱RAG系统 - 基础版", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚗 汽车座舱RAG系统 - 基础版")
        gr.Markdown("### ⚠️ 注意：当前运行在基础模式")
        gr.Markdown("由于模型下载问题，部分功能暂时不可用")
        
        with gr.Tab("📄 文档处理"):
            gr.Markdown("### 文档处理功能")
            gr.Markdown("当前模式下，文档处理功能受限")
            file_input = gr.File(label="上传车辆手册文件", file_types=[".pdf", ".txt", ".md", ".docx"])
            process_btn = gr.Button("处理文档", variant="primary")
            output_text = gr.Textbox(label="处理结果", lines=10)
            
            def process_file(file):
                if file is None:
                    return "请先上传文件"
                return f"文件 '{file.name}' 已接收。\n完整功能需要下载模型文件。\n请运行: python download_models.py"
            
            process_btn.click(process_file, inputs=[file_input], outputs=[output_text])
        
        with gr.Tab("❓ 问答"):
            gr.Markdown("### 智能问答")
            gr.Markdown("当前模式下，问答功能受限")
            question = gr.Textbox(label="输入问题", placeholder="例如：特斯拉Model 3的电池容量是多少？")
            ask_btn = gr.Button("提问", variant="primary")
            answer = gr.Textbox(label="回答", lines=10)
            
            def answer_question(query):
                if not query.strip():
                    return "请输入问题"
                return f"问题: '{query}'\n\n完整问答功能需要下载语言模型。\n请运行: python download_models.py"
            
            ask_btn.click(answer_question, inputs=[question], outputs=[answer])
        
        with gr.Tab("🔧 系统状态"):
            gr.Markdown("### 系统状态")
            gr.Markdown("""
            **当前状态**: 基础模式运行
            
            **缺失功能**:
            - 🤖 大语言模型推理
            - 🔍 向量检索
            - 🎤 语音识别
            - 🧠 意图分类
            
            **解决方案**:
            1. 运行 `python download_models.py` 下载模型
            2. 运行 `python test_model_load.py` 测试连接
            3. 运行 `python fix_huggingface.py` 修复连接
            4. 使用 `start_improved.bat` 启动
            
            **当前配置**:
            - Python版本: {sys.version}
            - 工作目录: {os.getcwd()}
            """)
        
        with gr.Tab("📖 使用说明"):
            gr.Markdown("### 使用说明")
            gr.Markdown("""
            ## 解决模型下载问题
            
            ### 方法1: 自动下载
            ```bash
            python download_models.py
            ```
            选择选项1下载必需的小模型
            
            ### 方法2: 修复连接
            ```bash
            python fix_huggingface.py
            ```
            选择选项1快速修复
            
            ### 方法3: 离线模式
            ```bash
            python offline_config.py
            ```
            按照说明手动下载模型
            
            ### 方法4: 使用代理
            ```bash
            set HTTP_PROXY=http://127.0.0.1:7890
            python main.py
            ```
            
            ## 完整功能启动
            解决下载问题后，使用:
            ```bash
            python main.py
            ```
            或
            ```bash
            start_improved.bat
            ```
            """)
    
    return demo

def main():
    """主函数"""
    print("🚗 汽车座舱RAG系统 - 简化版本")
    print("=" * 50)
    print("⚠️  注意: 由于HuggingFace连接问题，使用轻量级模型")
    print("=" * 50)
    
    # 检查依赖
    if not check_dependencies():
        print("\n❌ 缺少必要依赖，无法启动")
        print("   请运行: pip install -r requirements.txt")
        return
    
    # 检查数据目录
    print("\n📁 检查项目结构...")
    required_dirs = [
        "data/raw",
        "data/chunks", 
        "data/qa_train",
        "vector_store/faiss_index",
        "logs",
        "models_cache",
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ⚠️  {dir_path} - 创建中...")
            path.mkdir(parents=True, exist_ok=True)
    
    # 尝试加载小模型
    print("\n🔄 尝试加载轻量级模型...")
    try:
        # 测试是否能导入transformers
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        print("✅ 模型加载测试通过")
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)[:200]}")
        print("\n💡 运行以下命令解决问题:")
        print("   1. python fix_huggingface.py")
        print("   2. python download_models.py")
        print("   3. 或使用基础模式继续")
        
        choice = input("\n是否使用基础模式启动？(y/n): ").strip().lower()
        if choice != 'y':
            print("启动取消")
            return
    
    # 创建并启动应用
    print("\n🌐 启动Web界面...")
    print("=" * 50)
    
    try:
        app = create_simple_app()
    except Exception as e:
        print(f"❌ 创建应用失败，使用基础模式: {str(e)[:200]}")
        app = create_basic_app()
    
    # 获取本地IP
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "localhost"
    
    print(f"\n✅ 应用已启动!")
    print(f"   本地访问: http://localhost:7860")
    print(f"   网络访问: http://{local_ip}:7860")
    print(f"\n📱 当前模式: 简化版（部分功能受限）")
    print(f"💡 解决模型问题后，使用 main.py 启动完整版")
    
    # 启动应用
    print(f"\n🌐 启动Gradio服务器...")
    print(f"   本地访问: http://localhost:7860")
    print(f"   网络访问: http://{local_ip}:7860")
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
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 应用已关闭")
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        print("\n🔧 故障排除:")
        print("   1. 检查Python版本: python --version")
        print("   2. 安装依赖: pip install -r requirements.txt")
        print("   3. 修复连接: python fix_huggingface.py")
        print("   4. 下载模型: python download_models.py")
        input("\n按Enter键退出...")