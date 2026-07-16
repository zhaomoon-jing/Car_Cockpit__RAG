#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG LLM模块 - 基于检索增强生成的大语言模型推理
"""

import os
import json
import torch
import time
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import logging
from tqdm import tqdm

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, RAG_CONFIG
from retriever.dense_retriever import DenseRetriever
from retriever.bm25_retriever import BM25Retriever
from retriever.rerank import Reranker, HybridReranker
from intent_cls.infer_intent import IntentClassifier

# 模型下载配置
MODEL_DOWNLOAD_CONFIG = {
    "hf_mirrors": [
        "https://hf-mirror.com",
        "https://mirror.sjtu.edu.cn/huggingface",
        "https://huggingface.co",
    ],
    "timeout": 30,
    "max_retries": 3,
    "backoff_factor": 2.0,
    "use_local_cache": MODEL_CONFIG.get("model_download", {}).get("use_local_cache", False),
    "local_cache_path": MODEL_CONFIG.get("model_download", {}).get("local_cache_path", ""),
}

def setup_local_cache():
    """设置本地缓存环境"""
    if MODEL_DOWNLOAD_CONFIG["use_local_cache"] and MODEL_DOWNLOAD_CONFIG["local_cache_path"]:
        local_cache_path = Path(MODEL_DOWNLOAD_CONFIG["local_cache_path"])
        if local_cache_path.exists():
            # 设置HuggingFace缓存环境变量
            os.environ["HF_HOME"] = str(local_cache_path.parent)
            os.environ["TRANSFORMERS_CACHE"] = str(local_cache_path)
            os.environ["HF_DATASETS_CACHE"] = str(local_cache_path.parent / "datasets")
            os.environ["HF_MODULES_CACHE"] = str(local_cache_path.parent / "modules")
            os.environ["HF_HUB_OFFLINE"] = "1"  # 启用离线模式
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            
            logger = logging.getLogger(__name__)
            logger.info(f"✅ 使用本地缓存模式，缓存目录: {local_cache_path}")
            return True
        else:
            logger.warning(f"⚠️ 本地缓存目录不存在: {local_cache_path}")
    
    # 设置网络下载环境
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    return False

def get_local_model_path(model_name: str) -> Optional[str]:
    """获取模型在本地缓存中的路径"""
    _logger = logging.getLogger(__name__)
    
    if not MODEL_DOWNLOAD_CONFIG["use_local_cache"] or not MODEL_DOWNLOAD_CONFIG["local_cache_path"]:
        _logger.debug(f"[get_local_model_path] use_local_cache={MODEL_DOWNLOAD_CONFIG['use_local_cache']}, local_cache_path={MODEL_DOWNLOAD_CONFIG['local_cache_path']}")
        return None
    
    # 按优先级依次检查多个缓存路径
    cache_paths_to_check = []
    
    # 1. 配置中指定的缓存路径
    configured_path = Path(MODEL_DOWNLOAD_CONFIG["local_cache_path"])
    if configured_path.exists():
        cache_paths_to_check.append(configured_path)
    
    # 2. 项目内的 models_cache 目录（回退）
    project_cache = Path(__file__).parent.parent / "models_cache"
    if project_cache.exists() and project_cache != configured_path:
        cache_paths_to_check.append(project_cache)
    
    for local_cache_path in cache_paths_to_check:
        _logger.info(f"[get_local_model_path] 检查缓存路径: {local_cache_path}")
        
        # 将模型名称转换为缓存目录格式
        # 例如: Qwen/Qwen2.5-7B-Instruct -> models--Qwen--Qwen2.5-7B-Instruct
        cache_dir_name = model_name.replace("/", "--")
        if not cache_dir_name.startswith("models--"):
            cache_dir_name = f"models--{cache_dir_name}"
        
        model_cache_dir = local_cache_path / cache_dir_name
        if not model_cache_dir.exists():
            _logger.info(f"[get_local_model_path] 模型目录不存在: {model_cache_dir}")
            continue
        
        snapshots_dir = model_cache_dir / "snapshots"
        if not snapshots_dir.exists():
            _logger.info(f"[get_local_model_path] snapshots目录不存在: {snapshots_dir}")
            continue
        
        snapshots = list(snapshots_dir.iterdir())
        if not snapshots:
            _logger.info(f"[get_local_model_path] snapshots为空: {snapshots_dir}")
            continue
        
        # 使用最新的快照
        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
        _logger.info(f"[get_local_model_path] 找到快照: {latest_snapshot}")
        return str(latest_snapshot)
    
    _logger.info(f"[get_local_model_path] 所有缓存路径均未找到 {model_name} 的完整模型")
    return None

# 初始化本地缓存设置
setup_local_cache()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGLLM:
    """RAG LLM推理器"""
    
    def __init__(self, 
                 model_name: str = None,
                 lora_model_path: Optional[Path] = None,
                 device: str = None,
                 use_reranker: bool = True,
                 use_intent: bool = True):
        """
        初始化RAG LLM
        
        Args:
            model_name: 基础模型名称
            lora_model_path: LoRA模型路径
            device: 运行设备
            use_reranker: 是否使用重排序
            use_intent: 是否使用意图分类
        """
        self.model_name = model_name or MODEL_CONFIG['llm_model']
        self.lora_model_path = lora_model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_reranker = use_reranker
        self.use_intent = use_intent
        
        self.model = None
        self.tokenizer = None
        self.retriever = None
        self.reranker = None
        self.intent_classifier = None
        
        # RAG配置
        self.top_k = RAG_CONFIG['top_k']
        self.rerank_top_k = RAG_CONFIG['rerank_top_k']
        self.temperature = RAG_CONFIG['temperature']
        self.max_new_tokens = RAG_CONFIG['max_new_tokens']
        
        logger.info(f"初始化RAG LLM，模型: {self.model_name}，设备: {self.device}")
        
        # 加载组件
        self._load_components()
    
    def _load_components(self):
        """加载所有组件"""
        # 加载LLM模型
        self._load_llm()
        
        # 加载检索器
        self._load_retriever()
        
        # 加载重排序器
        if self.use_reranker:
            self._load_reranker()
        
        # 加载意图分类器
        if self.use_intent:
            self._load_intent_classifier()
    
    def _load_llm(self):
        """加载LLM模型（优先本地缓存，从小到大尝试，找不到再从镜像源下载）"""
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print(f"\n{'='*60}")
        print(f"[LLM] 开始加载LLM模型: {self.model_name}")
        print(f"[LLM] 目标设备: {self.device}")
        print(f"{'='*60}")
        logger.info(f"正在加载LLM模型: {self.model_name}")
        
        # ========== 第一步：优先检查本地缓存 ==========
        local_model_path = get_local_model_path(self.model_name)
        if local_model_path:
            print(f"[LLM] ✅ 找到本地缓存: {local_model_path}")
            print(f"[LLM] 📂 正在从本地加载: {self.model_name}")
            logger.info(f"✅ 找到本地缓存: {local_model_path}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    local_model_path, trust_remote_code=True, local_files_only=True
                )
                print(f"[LLM] ✅ Tokenizer加载成功")
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_model_path, torch_dtype=torch.float16,
                    low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True
                )
                print(f"[LLM] ✅ 模型权重加载成功 (float16)")
                
                if self.lora_model_path and self.lora_model_path.exists():
                    print(f"[LLM] 📂 加载LoRA: {self.lora_model_path}")
                    try:
                        from peft import PeftModel
                        self.model = PeftModel.from_pretrained(self.model, self.lora_model_path)
                        print(f"[LLM] ✅ LoRA加载成功")
                    except Exception as e:
                        print(f"[LLM] ⚠️ LoRA失败: {str(e)[:150]}")
                
                self.model = self.model.cuda() if self.device == "cuda" else self.model.cpu()
                print(f"[LLM] 📍 已移动到 {'GPU' if self.device == 'cuda' else 'CPU'}")
                self.model.eval()
                print(f"[LLM] 🎉 本地模型加载完成! {self.model_name}")
                print(f"[LLM] 📦 路径: {local_model_path}")
                print(f"{'='*60}\n")
                return
                
            except (OSError, MemoryError) as e:
                error_msg = str(e)
                if "页面文件太小" in error_msg or "1455" in error_msg or "out of memory" in error_msg.lower():
                    print(f"[LLM] ❌ 内存不足! {self.model_name} 太大，将尝试更小的模型...")
                    if hasattr(self, 'model') and self.model is not None: del self.model
                    self.model = None
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None: del self.tokenizer
                    self.tokenizer = None
                    import gc; gc.collect()
                    # OOM时不return，继续往下走备用小模型
                else:
                    print(f"[LLM] ⚠️ 本地加载失败: {error_msg[:200]}")
                    print(f"[LLM] 🔄 将尝试镜像源或备用模型...")
            except Exception as e:
                print(f"[LLM] ⚠️ 本地加载失败: {str(e)[:200]}")
        else:
            print(f"[LLM] ⚠️ 未找到 {self.model_name} 的本地缓存")
        
        # ========== 第二步：按从小到大依次尝试所有候选模型 ==========
        # 每个模型都先检查本地缓存，找不到再从镜像源下载
        all_models = [
            ("Qwen/Qwen2.5-0.5B-Instruct", "0.5B"),              # 最小，~1-2GB
            ("Qwen/Qwen2.5-1.5B-Instruct", "1.5B"),    # 小模型，~3GB
            ("microsoft/phi-2", "2.7B"),                # 小模型，~5GB
            ("THUDM/chatglm3-6b", "6B"),                # 中等，~12GB
            ("Qwen/Qwen2.5-7B-Instruct", "7B"),         # 大模型，~14GB
        ]
        # 过滤掉已经尝试过的主模型
        candidates = [(n, s) for n, s in all_models if n != self.model_name]
        print(f"\n[LLM] 📋 候选模型（从小到大）: {', '.join(f'{n}({s})' for n, s in candidates)}")
        
        for model_name, model_size in candidates:
            try:
                print(f"\n[LLM] --- 尝试: {model_name} ({model_size}) ---")
                self.model_name = model_name
                
                # 先检查本地缓存
                local_path = get_local_model_path(model_name)
                if local_path:
                    print(f"[LLM] ✅ 本地缓存: {local_path}")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        local_path, trust_remote_code=True, local_files_only=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        local_path, torch_dtype=torch.float16,
                        low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=True
                    )
                    print(f"[LLM] ✅ 从本地缓存加载成功")
                else:
                    # 本地没有，从镜像源下载
                    print(f"[LLM] 🌐 无本地缓存，从镜像源下载...")
                    downloaded = False
                    for mirror in MODEL_DOWNLOAD_CONFIG["hf_mirrors"]:
                        try:
                            os.environ["HF_ENDPOINT"] = mirror
                            print(f"[LLM] 🌐 尝试: {mirror}")
                            self.tokenizer = AutoTokenizer.from_pretrained(
                                model_name, trust_remote_code=True,
                                timeout=MODEL_DOWNLOAD_CONFIG["timeout"] * 2
                            )
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name, torch_dtype=torch.float16,
                                low_cpu_mem_usage=True, trust_remote_code=True,
                                timeout=MODEL_DOWNLOAD_CONFIG["timeout"] * 2
                            )
                            print(f"[LLM] ✅ 从 {mirror} 下载成功")
                            downloaded = True
                            break
                        except Exception as me:
                            print(f"[LLM] ⚠️ {mirror} 失败: {str(me)[:120]}")
                            continue
                    if not downloaded:
                        print(f"[LLM] ❌ 所有镜像源失败，跳过")
                        continue
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model = self.model.cuda() if self.device == "cuda" else self.model.cpu()
                print(f"[LLM] 📍 已移动到 {'GPU' if self.device == 'cuda' else 'CPU'}")
                self.model.eval()
                print(f"[LLM] 🎉 加载完成! {model_name} ({model_size})")
                print(f"{'='*60}\n")
                return
                
            except (OSError, MemoryError) as e:
                emsg = str(e)
                if "页面文件太小" in emsg or "1455" in emsg or "out of memory" in emsg.lower():
                    print(f"[LLM] ❌ {model_name} ({model_size}) 内存不足，尝试下一个...")
                    if hasattr(self, 'model') and self.model is not None: del self.model
                    self.model = None
                    if hasattr(self, 'tokenizer') and self.tokenizer is not None: del self.tokenizer
                    self.tokenizer = None
                    import gc; gc.collect()
                else:
                    print(f"[LLM] ⚠️ {model_name} ({model_size}) 失败: {emsg[:200]}")
            except Exception as e:
                print(f"[LLM] ⚠️ {model_name} ({model_size}) 失败: {str(e)[:200]}")
                continue
        
        print(f"\n[LLM] ❌ 无法加载任何LLM模型!")
        print(f"[LLM] 💡 请检查网络或手动下载模型到本地缓存")
        logger.error("无法加载任何LLM模型")
        raise ConnectionError("无法加载任何LLM模型。请检查网络或手动下载模型到本地缓存。")
    
    def _load_retriever(self):
        """加载检索器"""
        try:
            # 加载稠密检索器
            self.retriever = DenseRetriever()
            self.retriever.load_faiss_index()
            logger.info("稠密检索器加载成功")
            
            # 加载BM25检索器
            self.bm25_retriever = BM25Retriever()
            bm25_index_path = Path(__file__).parent.parent / "retriever" / "bm25_index" / "bm25_index.pkl"
            if bm25_index_path.exists():
                self.bm25_retriever.load_index(bm25_index_path)
                logger.info("BM25检索器加载成功")
            else:
                logger.warning("BM25索引未找到，将只使用稠密检索")
                self.bm25_retriever = None
            
        except Exception as e:
            logger.error(f"加载检索器失败: {str(e)}")
            raise
    
    def _load_reranker(self):
        """加载重排序器（CPU环境下跳过，太慢）"""
        if self.device == "cpu":
            logger.info("CPU环境下跳过重排序器（太慢），使用简单分数合并")
            self.reranker = None
            return
        try:
            self.reranker = HybridReranker()
            logger.info("重排序器加载成功")
        except Exception as e:
            logger.warning(f"加载重排序器失败: {str(e)}，将不使用重排序")
            self.reranker = None
    
    def _load_intent_classifier(self):
        """加载意图分类器"""
        try:
            intent_model_path = Path(__file__).parent.parent / "intent_cls" / "models"
            if intent_model_path.exists():
                self.intent_classifier = IntentClassifier(intent_model_path)
                logger.info("意图分类器加载成功")
            else:
                logger.warning("意图分类模型未找到，将不使用意图分类")
                self.intent_classifier = None
        except Exception as e:
            logger.warning(f"加载意图分类器失败: {str(e)}，将不使用意图分类")
            self.intent_classifier = None
    
    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[str, float, Dict]]: (文档内容, 分数, 元数据) 列表
        """
        top_k = top_k or self.top_k
        
        # 稠密检索（直接检索top_k，不做多余检索）
        dense_results = self.retriever.search_with_documents(query, top_k=top_k)
        
        # BM25检索（如果可用）
        bm25_results = []
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.search_with_documents(query, top_k=top_k)
        
        # 如果没有检索到结果，返回空列表
        if not dense_results and not bm25_results:
            return []
        
        # 如果只有一种检索结果，直接使用
        if not bm25_results:
            results = dense_results[:top_k]
        elif not dense_results:
            results = bm25_results[:top_k]
        else:
            # 混合检索和重排序
            if self.reranker:
                # 准备文档列表
                all_docs = []
                doc_index_map = {}
                
                # 收集所有文档
                for idx, (doc, score, meta) in enumerate(dense_results + bm25_results):
                    all_docs.append(doc)
                    doc_index_map[idx] = (score, meta)
                
                # 混合重排序
                hybrid_results = self.reranker.hybrid_rerank(
                    query=query,
                    dense_results=[(i, score, meta) for i, (_, score, meta) in enumerate(dense_results)],
                    bm25_results=[(i + len(dense_results), score, meta) for i, (_, score, meta) in enumerate(bm25_results)],
                    documents=all_docs,
                    top_k=top_k
                )
                
                results = [(doc, score, meta) for _, score, doc, meta in hybrid_results]
            else:
                # 简单合并和去重
                all_results = {}
                for doc, score, meta in dense_results:
                    doc_key = meta.get('chunk_id', doc[:100])
                    if doc_key not in all_results or score > all_results[doc_key][1]:
                        all_results[doc_key] = (doc, score, meta)
                
                for doc, score, meta in bm25_results:
                    doc_key = meta.get('chunk_id', doc[:100])
                    if doc_key not in all_results or score > all_results[doc_key][1]:
                        all_results[doc_key] = (doc, score, meta)
                
                # 按分数排序
                results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return results
    
    def analyze_intent(self, query: str) -> Dict[str, Any]:
        """
        分析查询意图
        
        Args:
            query: 查询文本
            
        Returns:
            Dict: 意图分析结果
        """
        if self.intent_classifier:
            return self.intent_classifier.analyze_query(query)
        else:
            return {
                "query": query,
                "primary_intent": {
                    "name": "通用查询",
                    "confidence": 1.0,
                    "description": "通用信息查询"
                },
                "all_intents": [],
                "suggested_actions": ["进行相关信息检索"]
            }
    
    def format_prompt(self, query: str, context: List[str], intent_info: Dict = None) -> str:
        """
        格式化prompt（精简版，减少输入token以加速生成）
        
        Args:
            query: 查询文本
            context: 相关文档列表
            intent_info: 意图信息
            
        Returns:
            str: 格式化后的prompt
        """
        # 精简系统提示（含领域限制）
        system_prompt = "你是汽车座舱助手，只回答与车辆使用、维护、故障诊断相关的问题。如果用户问的内容与汽车无关，或参考文档中没有相关信息，请明确告知无法回答。"
        
        # 上下文（精简，只保留关键内容）
        context_text = ""
        if context:
            context_text = "\n参考文档：\n"
            for i, doc in enumerate(context, 1):
                # 截断过长的文档内容，每个文档最多200字
                doc_trimmed = doc[:200] + "..." if len(doc) > 200 else doc
                context_text += f"{i}. {doc_trimmed}\n"
        
        # 用户查询
        prompt = f"{system_prompt}{context_text}\n用户问题：{query}\n回答："
        
        return prompt
    
    def _is_vehicle_related(self, query: str) -> bool:
        """检查查询是否与汽车/车辆相关"""
        vehicle_keywords = [
            "发动机", "轮胎", "刹车", "空调", "座椅", "方向盘", "变速箱", "离合",
            "电池", "充电", "续航", "电机", "车灯", "雨刷", "后视镜", "安全带",
            "气囊", "仪表", "导航", "音响", "屏幕", "车窗", "车门", "后备箱",
            "机油", "水箱", "散热", "排气", "悬挂", "转向", "挡位", "油门",
            "启动", "熄火", "换挡", "停车", "倒车", "加速", "减速", "巡航",
            "泊车", "解锁", "锁车", "开灯", "调温", "连接蓝牙",
            "车", "轿车", "SUV", "电车", "混动", "燃油", "新能源",
            "保养", "维修", "故障", "警告灯", "故障灯", "异响", "抖动",
            "胎压", "水温", "油耗", "配置", "型号",
            "公里", "转速", "扭矩", "马力", "排量",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in vehicle_keywords)
    
    def generate(self, query: str, context: List[str] = None, intent_info: Dict = None) -> str:
        """
        生成回答
        
        Args:
            query: 查询文本
            context: 相关文档列表（如果为None则自动检索）
            intent_info: 意图信息（如果为None则自动分析）
            
        Returns:
            str: 生成的回答
        """
        # 领域过滤：非汽车相关问题直接拒绝
        if not self._is_vehicle_related(query):
            logger.info(f"查询与汽车无关，已拒绝: {query}")
            return "抱歉，我是汽车座舱助手，只能回答与车辆使用、维护、故障诊断等相关的问题。您的问题不在我的服务范围内。"
        # 自动检索上下文
        if context is None:
            retrieved = self.retrieve(query)
            context = [doc for doc, _, _ in retrieved]
        
        # 自动分析意图
        if intent_info is None and self.use_intent:
            intent_info = self.analyze_intent(query)
        
        # 格式化prompt
        prompt = self.format_prompt(query, context, intent_info)
        
        # 生成回答
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码输出
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分（移除prompt）
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"生成回答失败: {str(e)}")
            return f"生成回答时出现错误: {str(e)}"
    
    def rag_pipeline(self, query: str) -> Dict[str, Any]:
        """
        完整的RAG pipeline
        
        Args:
            query: 查询文本
            
        Returns:
            Dict: 包含所有步骤结果
        """
        result = {
            "query": query,
            "timestamp": str(Path(__file__).parent.parent / "llm_infer" / "generated"),
            "steps": {}
        }
        
        # 步骤1: 意图分析
        if self.use_intent:
            intent_info = self.analyze_intent(query)
            result["intent_analysis"] = intent_info
            result["steps"]["intent_analysis"] = "completed"
        
        # 步骤2: 检索
        retrieved = self.retrieve(query)
        result["retrieval"] = {
            "total_results": len(retrieved),
            "results": [
                {
                    "content": doc[:200] + "..." if len(doc) > 200 else doc,
                    "score": float(score),
                    "metadata": meta
                }
                for doc, score, meta in retrieved
            ]
        }
        result["steps"]["retrieval"] = "completed"
        
        # 步骤3: 生成
        context = [doc for doc, _, _ in retrieved]
        intent_info = result.get("intent_analysis") if self.use_intent else None
        
        answer = self.generate(query, context, intent_info)
        result["generation"] = {
            "answer": answer,
            "context_used": len(context),
            "model": self.model_name
        }
        result["steps"]["generation"] = "completed"
        
        return result

def test_rag_pipeline():
    """测试RAG pipeline"""
    print("=" * 50)
    print("RAG Pipeline测试")
    print("=" * 50)
    
    # 初始化RAG LLM
    try:
        rag_llm = RAGLLM()
        print("✅ RAG LLM初始化成功")
    except Exception as e:
        print(f"❌ RAG LLM初始化失败: {str(e)}")
        return
    
    # 测试查询
    test_queries = [
        "特斯拉Model 3的电池容量是多少？",
        "车辆无法启动怎么办？",
        "怎么保养电动汽车？",
        "安全气囊注意事项有哪些？",
        "自动驾驶功能怎么开启？"
    ]
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 50)
        
        # 运行完整pipeline
        result = rag_llm.rag_pipeline(query)
        
        # 显示意图分析
        if "intent_analysis" in result:
            intent = result["intent_analysis"]["primary_intent"]
            print(f"🎯 意图: {intent['name']} ({intent['confidence']:.2%})")
        
        # 显示检索结果
        retrieval = result["retrieval"]
        print(f"📚 检索到 {retrieval['total_results']} 个相关文档:")
        for i, res in enumerate(retrieval["results"][:3], 1):
            source = res["metadata"].get("source", "未知")
            score = res["score"]
            print(f"  {i}. [{source}] (相关度: {score:.4f})")
            print(f"     内容: {res['content'][:100]}...")
        
        # 显示生成结果
        generation = result["generation"]
        print(f"\n💬 回答: {generation['answer']}")
        
        print("-" * 50)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='RAG LLM推理')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # query命令
    query_parser = subparsers.add_parser('query', help='查询问题')
    query_parser.add_argument('question', type=str, help='问题文本')
    query_parser.add_argument('--model', type=str, default=None, help='模型路径')
    query_parser.add_argument('--lora', type=str, default=None, help='LoRA模型路径')
    query_parser.add_argument('--no-rerank', action='store_true', help='不使用重排序')
    query_parser.add_argument('--no-intent', action='store_true', help='不使用意图分类')
    
    # pipeline命令
    pipeline_parser = subparsers.add_parser('pipeline', help='运行完整RAG pipeline')
    pipeline_parser.add_argument('question', type=str, help='问题文本')
    pipeline_parser.add_argument('--output', type=str, default='rag_result.json', help='输出文件路径')
    
    # test命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    
    # interactive命令
    interactive_parser = subparsers.add_parser('interactive', help='交互模式')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        test_rag_pipeline()
        return
    
    # 初始化RAG LLM
    try:
        lora_path = Path(args.lora) if args.lora else None
        rag_llm = RAGLLM(
            lora_model_path=lora_path,
            use_reranker=not args.no_rerank,
            use_intent=not args.no_intent
        )
        print("✅ RAG LLM初始化成功")
    except Exception as e:
        print(f"❌ RAG LLM初始化失败: {str(e)}")
        exit(1)
    
    if args.command == 'query':
        # 单条查询
        print(f"\n🔍 查询: '{args.question}'")
        
        answer = rag_llm.generate(args.question)
        
        print(f"\n💬 回答: {answer}")
    
    elif args.command == 'pipeline':
        # 完整pipeline
        print(f"\n🚀 运行RAG pipeline...")
        print(f"查询: '{args.question}'")
        
        result = rag_llm.rag_pipeline(args.question)
        
        # 保存结果
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Pipeline完成!")
        print(f"结果已保存到: {output_path}")
        
        # 显示摘要
        print(f"\n📊 摘要:")
        print(f"  意图: {result.get('intent_analysis', {}).get('primary_intent', {}).get('name', '未知')}")
        print(f"  检索文档数: {result.get('retrieval', {}).get('total_results', 0)}")
        print(f"  使用上下文数: {result.get('generation', {}).get('context_used', 0)}")
        print(f"  回答长度: {len(result.get('generation', {}).get('answer', ''))} 字符")
    
    elif args.command == 'interactive':
        # 交互模式
        print("RAG LLM - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        while True:
            try:
                question = input("\n请输入问题: ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not question:
                    continue
                
                print("\n🤔 思考中...")
                
                # 运行pipeline
                result = rag_llm.rag_pipeline(question)
                
                # 显示意图
                if "intent_analysis" in result:
                    intent = result["intent_analysis"]["primary_intent"]
                    print(f"\n🎯 识别意图: {intent['name']} ({intent['confidence']:.2%})")
                
                # 显示检索摘要
                retrieval = result["retrieval"]
                print(f"📚 参考了 {retrieval['total_results']} 个相关文档")
                
                # 显示回答
                generation = result["generation"]
                print(f"\n💬 回答: {generation['answer']}")
                
                # 显示来源（前3个）
                print(f"\n📖 参考来源:")
                for i, res in enumerate(retrieval["results"][:3], 1):
                    source = res["metadata"].get("source", "未知")
                    page = res["metadata"].get("page_number", "未知")
                    print(f"  {i}. {source} (第{page}页)")
                
            except KeyboardInterrupt:
                print("\n\n程序已终止")
                break
            except Exception as e:
                print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()