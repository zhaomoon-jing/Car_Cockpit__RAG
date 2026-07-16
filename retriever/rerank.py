#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重排序模块 - 对检索结果进行重新排序
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Reranker:
    """重排序器基类"""
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        
        logger.info(f"初始化重排序器，模型: {model_name}")
    
    def load_model(self):
        """加载重排序模型"""
        try:
            from FlagEmbedding import FlagReranker
            
            logger.info(f"正在加载重排序模型: {self.model_name}")
            self.model = FlagReranker(self.model_name, use_fp16=False)
            logger.info("重排序模型加载成功")
            
        except ImportError:
            logger.error("请安装 FlagEmbedding: pip install FlagEmbedding")
            raise
        except Exception as e:
            logger.error(f"加载重排序模型失败: {str(e)}")
            raise
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float, str]]:
        """
        对文档进行重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[int, float, str]]: (文档索引, 分数, 文档内容) 列表
        """
        if self.model is None:
            self.load_model()
        
        if not documents:
            return []
        
        # 准备查询-文档对
        pairs = [[query, doc] for doc in documents]
        
        # 计算分数
        logger.info(f"计算 {len(pairs)} 个查询-文档对的分数...")
        scores = self.model.compute_score(pairs)
        
        # 转换为列表（如果返回的是单个分数）
        if isinstance(scores, float):
            scores = [scores]
        
        # 排序
        sorted_indices = np.argsort(scores)[::-1]  # 降序排序
        
        # 选择top-k
        if top_k is not None:
            sorted_indices = sorted_indices[:top_k]
        
        # 构建结果
        results = []
        for idx in sorted_indices:
            if 0 <= idx < len(documents):
                results.append((idx, float(scores[idx]), documents[idx]))
        
        return results
    
    def rerank_with_metadata(self, query: str, documents_with_meta: List[Tuple[str, Dict]], top_k: int = None) -> List[Tuple[int, float, str, Dict]]:
        """
        对带元数据的文档进行重新排序
        
        Args:
            query: 查询文本
            documents_with_meta: (文档内容, 元数据) 列表
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[int, float, str, Dict]]: (文档索引, 分数, 文档内容, 元数据) 列表
        """
        documents = [doc for doc, _ in documents_with_meta]
        metadata_list = [meta for _, meta in documents_with_meta]
        
        # 重排序
        reranked = self.rerank(query, documents, top_k)
        
        # 添加元数据
        results = []
        for idx, score, doc in reranked:
            if 0 <= idx < len(metadata_list):
                results.append((idx, score, doc, metadata_list[idx]))
        
        return results

class HybridReranker:
    """混合重排序器（结合多种方法）"""
    
    def __init__(self, dense_weight: float = 0.7, bm25_weight: float = 0.2, rerank_weight: float = 0.1):
        """
        初始化混合重排序器
        
        Args:
            dense_weight: 稠密检索权重
            bm25_weight: BM25检索权重
            rerank_weight: 重排序权重
        """
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.rerank_weight = rerank_weight
        
        self.reranker = Reranker()
        
        logger.info(f"初始化混合重排序器，权重: 稠密={dense_weight}, BM25={bm25_weight}, 重排序={rerank_weight}")
    
    def hybrid_rerank(self, query: str, 
                     dense_results: List[Tuple[int, float, Dict]], 
                     bm25_results: List[Tuple[int, float, Dict]],
                     documents: List[str],
                     top_k: int = 5) -> List[Tuple[int, float, str, Dict]]:
        """
        混合重排序
        
        Args:
            query: 查询文本
            dense_results: 稠密检索结果 (索引, 分数, 元数据)
            bm25_results: BM25检索结果 (索引, 分数, 元数据)
            documents: 文档内容列表
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[int, float, str, Dict]]: 重排序结果
        """
        # 合并所有候选文档
        candidate_indices = set()
        
        # 添加稠密检索结果
        dense_scores = {}
        for idx, score, metadata in dense_results:
            candidate_indices.add(idx)
            dense_scores[idx] = score
        
        # 添加BM25结果
        bm25_scores = {}
        for idx, score, metadata in bm25_results:
            candidate_indices.add(idx)
            bm25_scores[idx] = score
        
        # 归一化分数
        def normalize_scores(score_dict):
            if not score_dict:
                return {}
            max_score = max(score_dict.values())
            min_score = min(score_dict.values())
            if max_score == min_score:
                return {k: 0.5 for k in score_dict.keys()}
            return {k: (v - min_score) / (max_score - min_score) for k, v in score_dict.items()}
        
        dense_scores_norm = normalize_scores(dense_scores)
        bm25_scores_norm = normalize_scores(bm25_scores)
        
        # 计算初始混合分数
        hybrid_scores = {}
        for idx in candidate_indices:
            dense_score = dense_scores_norm.get(idx, 0)
            bm25_score = bm25_scores_norm.get(idx, 0)
            hybrid_scores[idx] = (dense_score * self.dense_weight + 
                                 bm25_score * self.bm25_weight)
        
        # 选择top-2k进行重排序
        k_rerank = min(20, len(candidate_indices))  # 重排序候选数
        top_indices = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:k_rerank]
        top_indices = [idx for idx, _ in top_indices]
        
        # 获取候选文档内容
        candidate_docs = []
        candidate_metadata = []
        for idx in top_indices:
            if 0 <= idx < len(documents):
                candidate_docs.append(documents[idx])
                candidate_metadata.append(self._get_metadata_for_index(idx, dense_results, bm25_results))
        
        # 重排序
        reranked = self.reranker.rerank_with_metadata(query, 
                                                     list(zip(candidate_docs, candidate_metadata)), 
                                                     top_k=k_rerank)
        
        # 计算最终分数（混合分数 + 重排序分数）
        final_scores = {}
        for _, rerank_score, _, metadata in reranked:
            idx = metadata.get('index', -1)
            if idx >= 0:
                hybrid_score = hybrid_scores.get(idx, 0)
                final_scores[idx] = (hybrid_score * (1 - self.rerank_weight) + 
                                    rerank_score * self.rerank_weight)
        
        # 排序并返回结果
        sorted_indices = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_indices:
            if 0 <= idx < len(documents):
                doc = documents[idx]
                metadata = self._get_metadata_for_index(idx, dense_results, bm25_results)
                results.append((idx, score, doc, metadata))
        
        return results
    
    def _get_metadata_for_index(self, idx: int, 
                               dense_results: List[Tuple[int, float, Dict]], 
                               bm25_results: List[Tuple[int, float, Dict]]) -> Dict:
        """根据索引获取元数据"""
        # 首先从稠密检索结果中查找
        for dense_idx, _, metadata in dense_results:
            if dense_idx == idx:
                return metadata
        
        # 然后从BM25结果中查找
        for bm25_idx, _, metadata in bm25_results:
            if bm25_idx == idx:
                return metadata
        
        # 如果都没找到，返回默认元数据
        return {"index": idx, "source": "unknown"}

class DiversityReranker:
    """多样性重排序器（避免结果冗余）"""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        初始化多样性重排序器
        
        Args:
            similarity_threshold: 相似度阈值，高于此值的文档视为冗余
        """
        self.similarity_threshold = similarity_threshold
        
        # 加载句子嵌入模型用于计算相似度
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
        except ImportError:
            logger.warning("sentence-transformers未安装，使用简单文本相似度")
            self.embedding_model = None
    
    def rerank_for_diversity(self, results: List[Tuple[int, float, str, Dict]], 
                           top_k: int = 5) -> List[Tuple[int, float, str, Dict]]:
        """
        多样性重排序
        
        Args:
            results: 原始结果列表 (索引, 分数, 文档内容, 元数据)
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple]: 多样性重排序后的结果
        """
        if not results:
            return []
        
        selected = []
        remaining = results.copy()
        
        while len(selected) < top_k and remaining:
            # 选择分数最高的
            best_idx = max(range(len(remaining)), key=lambda i: remaining[i][1])
            best_result = remaining.pop(best_idx)
            selected.append(best_result)
            
            # 移除与已选文档相似的文档
            if self.embedding_model:
                # 使用嵌入模型计算相似度
                selected_text = best_result[2]
                selected_embedding = self.embedding_model.encode([selected_text])[0]
                
                to_remove = []
                for i, (_, _, doc, _) in enumerate(remaining):
                    doc_embedding = self.embedding_model.encode([doc])[0]
                    similarity = np.dot(selected_embedding, doc_embedding) / (
                        np.linalg.norm(selected_embedding) * np.linalg.norm(doc_embedding))
                    
                    if similarity > self.similarity_threshold:
                        to_remove.append(i)
                
                # 从后往前移除
                for i in reversed(to_remove):
                    remaining.pop(i)
            else:
                # 使用简单文本相似度（Jaccard相似度）
                selected_words = set(best_result[2].split())
                
                to_remove = []
                for i, (_, _, doc, _) in enumerate(remaining):
                    doc_words = set(doc.split())
                    if len(selected_words & doc_words) / max(len(selected_words | doc_words), 1) > self.similarity_threshold:
                        to_remove.append(i)
                
                # 从后往前移除
                for i in reversed(to_remove):
                    remaining.pop(i)
        
        return selected

def test_reranker():
    """测试重排序器"""
    print("=" * 50)
    print("重排序器测试")
    print("=" * 50)
    
    # 示例文档
    query = "特斯拉的电池容量是多少？"
    
    documents = [
        "特斯拉Model 3的电池容量为75千瓦时，续航里程可达600公里。",
        "电动汽车的电池容量通常从50到100千瓦时不等。",
        "特斯拉车辆使用锂离子电池技术，具有高能量密度。",
        "电池容量是衡量电动汽车续航能力的重要指标。",
        "Model 3的标准续航版电池容量为60千瓦时。",
        "长续航版Model 3的电池容量更大，达到82千瓦时。",
        "电池容量会影响车辆的充电时间和续航里程。",
        "特斯拉超级充电站可以在30分钟内充电至80%。",
        "电池保修政策为8年或16万公里。",
        "定期保养可以延长电池使用寿命。"
    ]
    
    # 初始化重排序器
    try:
        reranker = Reranker()
        print("✅ 重排序器初始化成功")
    except Exception as e:
        print(f"❌ 重排序器初始化失败: {str(e)}")
        return
    
    # 重排序
    print(f"\n🔍 查询: '{query}'")
    print(f"文档数: {len(documents)}")
    
    results = reranker.rerank(query, documents, top_k=5)
    
    print(f"\n📊 重排序结果 (前5个):")
    for i, (idx, score, doc) in enumerate(results):
        print(f"\n{i+1}. [分数: {score:.4f}]")
        print(f"   文档 {idx}: {doc}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='重排序器')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # rerank命令
    rerank_parser = subparsers.add_parser('rerank', help='重排序文档')
    rerank_parser.add_argument('query', type=str, help='查询文本')
    rerank_parser.add_argument('--documents', type=str, nargs='+', help='文档列表')
    rerank_parser.add_argument('--file', type=str, help='包含文档的文件路径（每行一个）')
    rerank_parser.add_argument('--topk', type=int, default=5, help='返回前k个结果')
    rerank_parser.add_argument('--model', type=str, default='BAAI/bge-reranker-base', help='重排序模型')
    
    # test命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        test_reranker()
        return
    
    # 加载文档
    documents = []
    if args.documents:
        documents = args.documents
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 文件不存在: {args.file}")
            exit(1)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            documents = [line.strip() for line in f if line.strip()]
    
    if not documents:
        print("❌ 没有文档可重排序")
        print("请通过 --documents 或 --file 参数提供文档")
        exit(1)
    
    # 初始化重排序器
    try:
        reranker = Reranker(model_name=args.model)
        print(f"✅ 重排序器初始化成功，模型: {args.model}")
    except Exception as e:
        print(f"❌ 重排序器初始化失败: {str(e)}")
        exit(1)
    
    # 重排序
    print(f"\n🔍 查询: '{args.query}'")
    print(f"文档数: {len(documents)}")
    
    results = reranker.rerank(args.query, documents, top_k=args.topk)
    
    print(f"\n📊 重排序结果 (前{args.topk}个):")
    for i, (idx, score, doc) in enumerate(results):
        print(f"\n{i+1}. [分数: {score:.4f}]")
        print(f"   文档 {idx}: {doc[:100]}...")

if __name__ == "__main__":
    main()