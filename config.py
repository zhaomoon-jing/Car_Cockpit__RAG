#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
汽车座舱RAG系统 - 全局配置文件
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent

# 数据目录
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"  # 原始PDF/TXT文件
CHUNKS_DIR = DATA_DIR / "chunks"  # 分块后的JSON文件
QA_TRAIN_DIR = DATA_DIR / "qa_train"  # LoRA微调QA数据集

# 数据处理目录
DATA_PROCESS_DIR = ROOT_DIR / "data_process"

# 向量存储目录
VECTOR_STORE_DIR = ROOT_DIR / "vector_store"
FAISS_INDEX_DIR = VECTOR_STORE_DIR / "faiss_index"  # FAISS索引文件

# 语音识别目录
SPEECH_ASR_DIR = ROOT_DIR / "speech_asr"

# 意图分类目录
INTENT_CLS_DIR = ROOT_DIR / "intent_cls"

# 检索器目录
RETRIEVER_DIR = ROOT_DIR / "retriever"

# LLM推理目录
LLM_INFER_DIR = ROOT_DIR / "llm_infer"

# Gradio Web目录
GRADIO_WEB_DIR = ROOT_DIR / "gradio_web"

# 确保所有目录存在
for dir_path in [
    RAW_DATA_DIR, CHUNKS_DIR, QA_TRAIN_DIR,
    FAISS_INDEX_DIR,
    DATA_PROCESS_DIR, VECTOR_STORE_DIR, SPEECH_ASR_DIR,
    INTENT_CLS_DIR, RETRIEVER_DIR, LLM_INFER_DIR, GRADIO_WEB_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_CONFIG = {
    # 文本嵌入模型
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "embedding_dim": 512,
    "llm_model": "Qwen/Qwen2.5-0.5B-Instruct",
    # LLM模型（主模型和备用模型）
    #由于内存问题使用0.6 B作为主模型
    #"llm_model": "Qwen/Qwen2.5-7B-Instruct",
    "llm_backup_model": "Qwen/Qwen2.5-0.5B-Instruct",  # 更小的备用模型
    "llm_max_length": 4096,
    "use_backup_llm": False,  # 是否使用备用模型
    
    # 语音识别模型
    "asr_model": "openai/whisper-small",
    "asr_backup_model": "openai/whisper-tiny",  # 更小的备用模型
    
    # 意图分类模型
    "intent_model": "BAAI/bge-small-zh-v1.5",
    "intent_backup_model": "hfl/chinese-roberta-wwm-ext",
    
    # LoRA配置
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    
    # 模型下载设置
    "model_download": {
        "timeout": 30,
        "max_retries": 3,
        "use_mirror": True,
        "mirror_url": "https://hf-mirror.com",
        "cache_dir": "models_cache",
        "use_local_cache": True,  # 启用本地缓存
        "local_cache_path": "F:/ModelCache/huggingface/hub",  # 本地缓存路径
    }
}

# RAG配置
RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k": 3,           # 从5减到3，减少检索量和prompt长度
    "rerank_top_k": 2,    # 从3减到2，减少重排序开销
    "temperature": 0.7,
    "max_new_tokens": 150,  # 从512减到150，大幅减少生成时间
}

# 数据库配置
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "database": "car_cockpit_rag",
    "username": "root",
    "password": "password",
}

# 日志配置
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": ROOT_DIR / "logs" / "car_cockpit_rag.log",
}

# 创建日志目录
(ROOT_DIR / "logs").mkdir(exist_ok=True)

if __name__ == "__main__":
    print("配置加载成功！")
    print(f"项目根目录: {ROOT_DIR}")
    print(f"数据目录: {DATA_DIR}")
    print(f"向量存储目录: {VECTOR_STORE_DIR}")