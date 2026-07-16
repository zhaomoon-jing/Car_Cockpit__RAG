#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Web界面 - 汽车座舱RAG系统可视化界面
"""

import os
import json
import gradio as gr
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import tempfile
import webbrowser

# 导入项目配置和模块
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import ROOT_DIR, RAW_DATA_DIR, CHUNKS_DIR, FAISS_INDEX_DIR
from data_process.parse_pdf import PDFParser, process_all_documents
from data_process.clean_chunk import TextCleaner, process_all_chunks as clean_all_chunks
from data_process.build_chunk import TextChunker, process_all_documents as chunk_all_documents
from vector_store.build_faiss import FAISSIndexBuilder
from speech_asr.whisper_asr import WhisperASR, AudioPreprocessor
from intent_cls.infer_intent import IntentClassifier
from retriever.dense_retriever import DenseRetriever
from retriever.bm25_retriever import BM25Retriever
from llm_infer.rag_llm import RAGLLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CarCockpitRAGApp:
    """汽车座舱RAG应用"""
    
    def __init__(self):
        self.rag_llm = None
        self.intent_classifier = None
        self.dense_retriever = None
        self.bm25_retriever = None
        self.asr = None
        
        # 状态
        self.is_initialized = False
        self.current_context = []
        
        # 初始化轻量组件（不加载大模型）
        self._init_components()
    
    def _init_components(self):
        """初始化组件（仅检查资源，不加载大模型，避免启动时OOM）"""
        try:
            # 检查必要目录
            self._check_directories()
            
            # 检查意图分类器（不加载模型，仅检查文件）
            intent_model_path = ROOT_DIR / "intent_cls" / "models"
            if intent_model_path.exists() and (intent_model_path / "config.json").exists():
                logger.info("意图分类模型已就绪（将在首次使用时加载）")
            else:
                logger.warning("意图分类模型未找到，将不使用意图分类")
            
            # 检查检索器（不加载模型，仅检查索引文件）
            if (FAISS_INDEX_DIR / "faiss_index.bin").exists():
                logger.info("FAISS索引已就绪（将在首次使用时加载）")
            else:
                logger.warning("FAISS索引未找到，请先运行向量化流程")
            
            bm25_index_path = ROOT_DIR / "retriever" / "bm25_index" / "bm25_index.pkl"
            if bm25_index_path.exists():
                logger.info("BM25索引已就绪")
            
            self.is_initialized = True
            logger.info("应用初始化完成（大模型将按需懒加载）")
            
        except Exception as e:
            logger.error(f"应用初始化失败: {str(e)}")
            self.is_initialized = False
    
    def _get_rag_llm(self):
        """懒加载RAG LLM"""
        if self.rag_llm is None:
            logger.info("首次使用，正在加载RAG LLM...")
            try:
                self.rag_llm = RAGLLM(
                    use_reranker=True,
                    use_intent=self.intent_classifier is not None
                )
            except Exception as e:
                logger.error(f"RAG LLM加载失败: {str(e)}")
                raise
        return self.rag_llm
    
    def _get_intent_classifier(self):
        """懒加载意图分类器"""
        if self.intent_classifier is None:
            intent_model_path = ROOT_DIR / "intent_cls" / "models"
            if intent_model_path.exists() and (intent_model_path / "config.json").exists():
                logger.info("正在加载意图分类器...")
                try:
                    self.intent_classifier = IntentClassifier(intent_model_path)
                except Exception as e:
                    logger.error(f"意图分类器加载失败: {str(e)}")
        return self.intent_classifier
    
    def _get_dense_retriever(self):
        """懒加载稠密检索器"""
        if self.dense_retriever is None:
            if (FAISS_INDEX_DIR / "faiss_index.bin").exists():
                logger.info("正在加载稠密检索器...")
                try:
                    self.dense_retriever = DenseRetriever()
                    self.dense_retriever.load_faiss_index()
                except Exception as e:
                    logger.error(f"稠密检索器加载失败: {str(e)}")
        return self.dense_retriever
    
    def _get_asr(self):
        """懒加载语音识别"""
        if self.asr is None:
            logger.info("正在加载语音识别模型...")
            try:
                self.asr = WhisperASR()
            except Exception as e:
                logger.error(f"语音识别模型加载失败: {str(e)}")
                raise
        return self.asr
    
    def _check_directories(self):
        """检查必要目录"""
        required_dirs = [RAW_DATA_DIR, CHUNKS_DIR, FAISS_INDEX_DIR]
        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        处理上传的文档
        
        Args:
            file_path: 文件路径
            
        Returns:
            Dict: 处理结果
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return {"success": False, "message": f"文件不存在: {file_path}"}
            
            # 复制文件到raw目录
            import shutil
            dest_path = RAW_DATA_DIR / file_path.name
            shutil.copy2(file_path, dest_path)
            
            return {
                "success": True,
                "message": f"文件已上传: {file_path.name}",
                "path": str(dest_path)
            }
            
        except Exception as e:
            logger.error(f"处理文档失败: {str(e)}")
            return {"success": False, "message": f"处理失败: {str(e)}"}
    
    def run_data_pipeline(self) -> Dict[str, Any]:
        """
        运行数据预处理流水线
        
        Returns:
            Dict: 处理结果
        """
        try:
            results = {
                "parse_pdf": {"success": False, "message": ""},
                "clean_chunk": {"success": False, "message": ""},
                "build_chunk": {"success": False, "message": ""},
                "build_faiss": {"success": False, "message": ""}
            }
            
            # 1. 解析PDF
            logger.info("开始解析PDF...")
            process_all_documents()
            results["parse_pdf"]["success"] = True
            results["parse_pdf"]["message"] = "PDF解析完成"
            
            # 2. 清洗文本
            logger.info("开始清洗文本...")
            clean_all_chunks()
            results["clean_chunk"]["success"] = True
            results["clean_chunk"]["message"] = "文本清洗完成"
            
            # 3. 分块处理
            logger.info("开始分块处理...")
            chunk_all_documents()
            results["build_chunk"]["success"] = True
            results["build_chunk"]["message"] = "文本分块完成"
            
            # 4. 构建FAISS索引
            logger.info("开始构建FAISS索引...")
            builder = FAISSIndexBuilder()
            builder.build_from_chunks()
            results["build_faiss"]["success"] = True
            results["build_faiss"]["message"] = "FAISS索引构建完成"
            
            # 重新初始化检索器
            self._init_components()
            
            return {
                "success": True,
                "message": "数据预处理流水线完成",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"数据预处理流水线失败: {str(e)}")
            return {
                "success": False,
                "message": f"处理失败: {str(e)}",
                "results": results
            }
    
    def transcribe_audio(self, audio_file: str) -> Dict[str, Any]:
        """
        转录音频文件
        """
        try:
            audio_path = Path(audio_file)
            
            # 验证音频文件
            preprocessor = AudioPreprocessor()
            is_valid, message = preprocessor.validate_audio_file(audio_path)
            
            if not is_valid:
                return {"success": False, "message": message}
            
            # 懒加载ASR并转录
            asr = self._get_asr()
            result = asr.transcribe_audio(audio_path)
            
            # 保存转录结果
            output_file = ROOT_DIR / "speech_asr" / f"{audio_path.stem}_transcription.json"
            self.asr.save_transcription(result, output_file)
            
            return {
                "success": True,
                "message": "音频转录完成",
                "text": result.get("text", ""),
                "language": result.get("language", ""),
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"音频转录失败: {str(e)}")
            return {"success": False, "message": f"转录失败: {str(e)}"}
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询意图
        """
        classifier = self._get_intent_classifier()
        if not classifier:
            return {
                "success": False,
                "message": "意图分类器未初始化",
                "intent": "未知",
                "confidence": 0.0
            }
        
        try:
            results = classifier.predict(query, top_k=3)
            
            if not results:
                return {
                    "success": False,
                    "message": "无法分析意图",
                    "intent": "未知",
                    "confidence": 0.0
                }
            
            intent, confidence, description = results[0]
            
            return {
                "success": True,
                "message": "意图分析完成",
                "intent": intent,
                "confidence": confidence,
                "description": description,
                "all_intents": [
                    {"name": i, "confidence": c, "description": d}
                    for i, c, d in results
                ]
            }
            
        except Exception as e:
            logger.error(f"意图分析失败: {str(e)}")
            return {
                "success": False,
                "message": f"分析失败: {str(e)}",
                "intent": "未知",
                "confidence": 0.0
            }
    
    def retrieve_documents(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        检索相关文档
        """
        retriever = self._get_dense_retriever()
        if not retriever:
            return {
                "success": False,
                "message": "检索器未初始化",
                "results": []
            }
        
        try:
            results = retriever.search_with_documents(query, top_k=top_k)
            
            formatted_results = []
            for doc, score, metadata in results:
                formatted_results.append({
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "score": float(score),
                    "source": metadata.get("source", "未知"),
                    "page": metadata.get("page_number", "未知"),
                    "metadata": metadata
                })
            
            return {
                "success": True,
                "message": f"检索到 {len(results)} 个相关文档",
                "results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            return {
                "success": False,
                "message": f"检索失败: {str(e)}",
                "results": []
            }
    
    def generate_answer(self, query: str, use_context: bool = True) -> Dict[str, Any]:
        """
        生成回答
        """
        try:
            rag_llm = self._get_rag_llm()
        except Exception as e:
            return {
                "success": False,
                "message": f"RAG LLM加载失败: {str(e)}",
                "answer": f"模型加载失败，请检查模型是否已下载: {str(e)}"
            }
        
        try:
            if use_context:
                result = rag_llm.rag_pipeline(query)
                
                return {
                    "success": True,
                    "message": "回答生成完成",
                    "answer": result["generation"]["answer"],
                    "intent": result.get("intent_analysis", {}).get("primary_intent", {}).get("name", "未知"),
                    "confidence": result.get("intent_analysis", {}).get("primary_intent", {}).get("confidence", 0.0),
                    "retrieved_docs": len(result["retrieval"]["results"]),
                    "context_used": result["generation"]["context_used"],
                    "full_result": result
                }
            else:
                answer = rag_llm.generate(query, context=[])
                
                return {
                    "success": True,
                    "message": "回答生成完成",
                    "answer": answer,
                    "intent": "未知",
                    "confidence": 0.0,
                    "retrieved_docs": 0,
                    "context_used": 0
                }
                
        except Exception as e:
            logger.error(f"回答生成失败: {str(e)}")
            return {
                "success": False,
                "message": f"生成失败: {str(e)}",
                "answer": f"生成回答时出现错误: {str(e)}"
            }
    
    def chat(self, query: str, history: List[List[str]]) -> List[List[str]]:
        """
        聊天接口
        
        Args:
            query: 用户查询
            history: 聊天历史
            
        Returns:
            List[List[str]]: 更新后的聊天历史
        """
        if not query.strip():
            return history
        
        # 生成回答
        result = self.generate_answer(query, use_context=True)
        
        if result["success"]:
            answer = result["answer"]
            # 添加意图信息
            if result["intent"] != "未知":
                answer = f"[意图: {result['intent']}]\n\n{answer}"
        else:
            answer = result["answer"]
        
        # 更新历史 (Gradio 4.x 默认 tuples 格式)
        history.append([query, answer])
        
        return history

def create_app():
    """创建Gradio应用"""
    app = CarCockpitRAGApp()
    
    with gr.Blocks(title="汽车座舱RAG系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚗 汽车座舱RAG系统")
        gr.Markdown("基于检索增强生成技术的汽车座舱智能助手")
        
        with gr.Tabs():
            # 标签页1: 文档处理
            with gr.Tab("📄 文档处理"):
                gr.Markdown("### 上传和处理车辆手册文档")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        file_input = gr.File(
                            label="上传车辆手册",
                            file_types=[".pdf", ".txt", ".docx", ".md"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("📤 上传文档", variant="primary")
                    
                    with gr.Column(scale=1):
                        upload_status = gr.Textbox(label="上传状态", interactive=False)
                
                gr.Markdown("### 数据预处理流水线")
                pipeline_btn = gr.Button("🔄 运行数据处理流水线", variant="secondary")
                pipeline_output = gr.Textbox(label="处理结果", lines=8, interactive=False)
            
            # 标签页2: 语音转录
            with gr.Tab("🎤 语音转录"):
                gr.Markdown("### 语音转文本")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        audio_input = gr.Audio(
                            label="上传音频文件",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        transcribe_btn = gr.Button("🎵 开始转录", variant="primary")
                    
                    with gr.Column(scale=1):
                        transcribe_status = gr.Textbox(label="转录状态", interactive=False)
                
                with gr.Row():
                    transcription_output = gr.Textbox(
                        label="转录文本",
                        lines=10,
                        max_lines=20,
                        interactive=False
                    )
            
            # 标签页3: 意图分析
            with gr.Tab("🎯 意图分析"):
                gr.Markdown("### 分析用户查询意图")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        intent_query = gr.Textbox(
                            label="输入查询",
                            placeholder="例如：特斯拉的电池容量是多少？",
                            lines=3
                        )
                        analyze_btn = gr.Button("🔍 分析意图", variant="primary")
                    
                    with gr.Column(scale=1):
                        intent_result = gr.Label(label="主要意图")
                        confidence_gauge = gr.Label(label="置信度")
                
                with gr.Row():
                    intent_details = gr.Textbox(label="详细结果", lines=6, interactive=False)
            
            # 标签页4: 文档检索
            with gr.Tab("📚 文档检索"):
                gr.Markdown("### 检索相关文档")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        retrieve_query = gr.Textbox(
                            label="检索查询",
                            placeholder="例如：车辆保养注意事项",
                            lines=2
                        )
                        retrieve_btn = gr.Button("🔎 检索文档", variant="primary")
                    
                    with gr.Column(scale=1):
                        retrieve_count = gr.Number(label="检索结果数", value=0, interactive=False)
                
                with gr.Row():
                    retrieve_results = gr.Textbox(label="检索结果", lines=8, interactive=False)
            
            # 标签页5: 智能问答
            with gr.Tab("💬 智能问答"):
                gr.Markdown("### 与车辆手册对话")
                
                chatbot = gr.Chatbot(
                    label="汽车座舱助手",
                    height=400,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="输入问题",
                        placeholder="请输入关于车辆的问题...",
                        scale=4
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
                    clear_btn = gr.Button("清空", variant="secondary", scale=1)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        use_context = gr.Checkbox(label="使用上下文检索", value=True)
                    with gr.Column(scale=1):
                        show_intent = gr.Checkbox(label="显示意图分析", value=True)
                
                gr.Markdown("### 示例问题")
                examples = gr.Examples(
                    examples=[
                        ["特斯拉Model 3的电池容量是多少？"],
                        ["车辆无法启动怎么办？"],
                        ["怎么保养电动汽车？"],
                        ["安全气囊注意事项有哪些？"],
                        ["自动驾驶功能怎么开启？"]
                    ],
                    inputs=msg,
                    label="点击示例快速提问"
                )
            

        
        # 事件处理
        # 上传文档
        def handle_upload(files):
            if not files:
                return "请选择文件"
            
            results = []
            for file in files:
                # Gradio 4.x: file 可能是字符串路径或 UploadFile 对象
                file_path = file.name if hasattr(file, 'name') else str(file)
                result = app.process_document(file_path)
                results.append(f"{Path(file_path).name}: {'成功' if result['success'] else '失败'} - {result['message']}")
            
            return "\n".join(results)
        
        upload_btn.click(
            handle_upload,
            inputs=[file_input],
            outputs=[upload_status]
        )
        
        # 运行数据处理流水线
        def _handle_pipeline():
            result = app.run_data_pipeline()
            return json.dumps(result, ensure_ascii=False, indent=2)
        
        pipeline_btn.click(
            _handle_pipeline,
            inputs=[],
            outputs=[pipeline_output]
        )
        
        # 语音转录
        def handle_transcribe(audio_file):
            try:
                if audio_file is None:
                    return "请上传音频文件", ""
                
                result = app.transcribe_audio(audio_file)
                if result["success"]:
                    return result["message"], result["text"]
                else:
                    return result["message"], ""
            except Exception as e:
                logger.error(f"语音转录异常: {str(e)}", exc_info=True)
                return f"转录失败: {str(e)}", ""
        
        transcribe_btn.click(
            handle_transcribe,
            inputs=[audio_input],
            outputs=[transcribe_status, transcription_output]
        )
        
        # 意图分析
        def handle_analyze(query):
            try:
                if not query.strip():
                    return "未知", "0%", "{}"
                
                result = app.analyze_query(query)
                if result["success"]:
                    intent = result["intent"]
                    confidence = f"{result['confidence']:.2%}"
                    details = json.dumps({
                        "intent": intent,
                        "confidence": result["confidence"],
                        "description": result["description"],
                        "all_intents": result.get("all_intents", [])
                    }, ensure_ascii=False, indent=2)
                    return intent, confidence, details
                else:
                    return "未知", "0%", json.dumps({"error": result["message"]}, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"意图分析异常: {str(e)}", exc_info=True)
                return "未知", "0%", json.dumps({"error": f"分析失败: {str(e)}"}, ensure_ascii=False, indent=2)
        
        analyze_btn.click(
            handle_analyze,
            inputs=[intent_query],
            outputs=[intent_result, confidence_gauge, intent_details]
        )
        
        # 文档检索
        def handle_retrieve(query):
            try:
                if not query.strip():
                    return 0, "[]"
                
                result = app.retrieve_documents(query, top_k=5)
                if result["success"]:
                    return len(result["results"]), json.dumps(result["results"], ensure_ascii=False, indent=2)
                else:
                    return 0, json.dumps([{"error": result["message"]}], ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"文档检索异常: {str(e)}", exc_info=True)
                return 0, json.dumps([{"error": f"检索失败: {str(e)}"}], ensure_ascii=False, indent=2)
        
        retrieve_btn.click(
            handle_retrieve,
            inputs=[retrieve_query],
            outputs=[retrieve_count, retrieve_results]
        )
        
        # 智能问答
        def handle_chat(message, history, use_context_flag, show_intent_flag):
            try:
                if not message.strip():
                    return "", history
                
                # 生成回答
                result = app.generate_answer(message, use_context=use_context_flag)
                
                if result["success"]:
                    answer = result["answer"]
                    if show_intent_flag and result["intent"] != "未知":
                        answer = f"**[意图: {result['intent']} ({result['confidence']:.2%})]**\n\n{answer}"
                else:
                    answer = result["answer"]
                
                history.append([message, answer])
                return "", history
            except Exception as e:
                logger.error(f"智能问答异常: {str(e)}", exc_info=True)
                history.append([message, f"抱歉，生成回答时出错: {str(e)}"])
                return "", history
        
        submit_btn.click(
            handle_chat,
            inputs=[msg, chatbot, use_context, show_intent],
            outputs=[msg, chatbot]
        ).then(
            lambda: gr.update(interactive=True),
            outputs=[msg]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            inputs=[],
            outputs=[chatbot, msg]
        )
    
    return demo

def main():
    """主函数"""
    print("=" * 50)
    print("🚗 汽车座舱RAG系统 - Gradio Web界面")
    print("=" * 50)
    
    # 创建应用
    demo = create_app()
    
    # 启动服务器
    print("\n🌐 启动Gradio服务器...")
    print("本地访问: http://localhost:7860")
    print("网络访问: 请查看Gradio提供的分享链接")
    print("\n按 Ctrl+C 停止服务器")
    
    # 获取本地IP地址
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"本地IP: http://{local_ip}:7860")
    except:
        pass
    
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()