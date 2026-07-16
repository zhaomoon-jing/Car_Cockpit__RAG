#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稠密检索器 - 基于向量相似度的检索
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from tqdm import tqdm
import faiss

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FAISS_INDEX_DIR, MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DenseRetriever:
    """稠密检索器（基于向量相似度）"""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        初始化稠密检索器
        
        Args:
            model_name: 嵌入模型名称
            device: 运行设备 ("cpu" 或 "cuda")
        """
        self.model_name = model_name or MODEL_CONFIG['embedding_model']
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.index = None
        self.metadata = []
        self.embedding_dim = MODEL_CONFIG['embedding_dim']
        
        logger.info(f"初始化DenseRetriever，模型: {self.model_name}，设备: {device}")
    
    def load_model(self):
        """加载嵌入模型（优先使用本地缓存）"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"正在加载模型: {self.model_name}")
            
            # 尝试使用本地缓存
            from local_model_cache import load_model_with_cache
            try:
                self.model = load_model_with_cache(
                    self.model_name,
                    SentenceTransformer,
                    device=self.device
                )
            except ImportError:
                # 如果local_model_cache不可用，直接加载
                self.model = SentenceTransformer(self.model_name, device=self.device)
            
            # 测试模型
            test_embedding = self.model.encode(["测试文本"])
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.embedding_dim:
                logger.warning(f"模型维度 ({actual_dim}) 与配置维度 ({self.embedding_dim}) 不匹配，使用实际维度")
                self.embedding_dim = actual_dim
            
            logger.info(f"模型加载成功，维度: {self.embedding_dim}")
            
        except ImportError:
            logger.error("请安装 sentence-transformers: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        编码查询文本为向量
        
        Args:
            query: 查询文本
            
        Returns:
            np.ndarray: 查询向量
        """
        if self.model is None:
            self.load_model()
        
        # 编码查询
        query_embedding = self.model.encode([query], show_progress_bar=False)
        
        # 归一化（对余弦相似度有益）
        faiss.normalize_L2(query_embedding)
        
        return query_embedding.astype('float32')
    
    def load_faiss_index(self, index_dir: Path = FAISS_INDEX_DIR):
        """
        加载FAISS索引
        
        Args:
            index_dir: 索引目录
        """
        index_file = index_dir / "faiss_index.bin"
        metadata_file = index_dir / "metadata.pkl"
        config_file = index_dir / "config.json"
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS索引文件不存在: {index_file}")
        
        # 加载FAISS索引
        logger.info(f"加载FAISS索引: {index_file}")
        self.index = faiss.read_index(str(index_file))
        
        # 加载元数据
        if metadata_file.exists():
            with open(metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            logger.info(f"加载元数据: {len(self.metadata)} 条")
        else:
            logger.warning(f"元数据文件不存在: {metadata_file}")
            self.metadata = [{} for _ in range(self.index.ntotal)]
        
        # 加载配置
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.embedding_dim = config.get('embedding_dim', self.embedding_dim)
            logger.info(f"加载配置: {config_file}")
        
        logger.info(f"FAISS索引加载完成，包含 {self.index.ntotal} 个向量")
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[int, float, Dict]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            threshold: 相似度阈值
            
        Returns:
            List[Tuple[int, float, Dict]]: (文档索引, 分数, 元数据) 列表
        """
        if self.index is None:
            raise RuntimeError("请先调用 load_faiss_index() 加载索引")
        
        # 编码查询
        query_embedding = self.encode_query(query)
        
        # 搜索
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 转换距离为相似度分数（余弦相似度）
        # FAISS使用内积，归一化后内积=余弦相似度
        scores = distances[0]
        
        results = []
        for idx, score in zip(indices[0], scores):
            if idx >= 0 and score > threshold:  # idx=-1表示未找到
                results.append((idx, float(score), self.metadata[idx]))
        
        return results
    
    def search_with_documents(self, query: str, top_k: int = 5, threshold: float = 0.0) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似文档并返回文档内容
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            threshold: 相似度阈值
            
        Returns:
            List[Tuple[str, float, Dict]]: (文档内容, 分数, 元数据) 列表
        """
        # 需要先加载文档内容
        # 这里假设metadata中包含文档内容或可以从中获取
        results = self.search(query, top_k, threshold)
        
        # 在实际应用中，需要根据metadata获取文档内容
        # 这里返回metadata中的信息
        return [
            (self._get_document_content(idx), score, metadata)
            for idx, score, metadata in results
        ]
    
    def _get_document_content(self, idx: int) -> str:
        """根据索引获取文档内容"""
        if idx < len(self.metadata):
            metadata = self.metadata[idx]
            # 这里可以根据实际存储方式获取文档内容
            # 例如从chunk_id加载文档
            return metadata.get('text', f"文档 {idx} 的内容未存储")
        return f"文档 {idx} 的元数据不存在"
    
    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[Tuple[int, float, Dict]]]:
        """
        批量搜索
        
        Args:
            queries: 查询文本列表
            top_k: 返回前k个结果
            
        Returns:
            List[List[Tuple]]: 每个查询的结果列表
        """
        if self.index is None:
            raise RuntimeError("请先调用 load_faiss_index() 加载索引")
        
        # 批量编码查询
        query_embeddings = []
        for query in tqdm(queries, desc="编码查询"):
            embedding = self.encode_query(query)
            query_embeddings.append(embedding)
        
        # 堆叠所有查询向量
        all_embeddings = np.vstack(query_embeddings)
        
        # 批量搜索
        distances, indices = self.index.search(all_embeddings, top_k)
        
        # 构建结果
        all_results = []
        for i in range(len(queries)):
            query_results = []
            for idx, score in zip(indices[i], distances[i]):
                if idx >= 0:
                    query_results.append((idx, float(score), self.metadata[idx]))
            all_results.append(query_results)
        
        return all_results
    
    def hybrid_search(self, query: str, bm25_retriever, top_k: int = 10, alpha: float = 0.5) -> List[Tuple[int, float, Dict]]:
        """
        混合检索（稠密检索 + BM25）
        
        Args:
            query: 查询文本
            bm25_retriever: BM25检索器实例
            top_k: 返回前k个结果
            alpha: 稠密检索权重 (1-alpha为BM25权重)
            
        Returns:
            List[Tuple[int, float, Dict]]: 混合检索结果
        """
        # 稠密检索
        dense_results = self.search(query, top_k=top_k * 2)
        
        # BM25检索
        bm25_results = bm25_retriever.search(query, top_k=top_k * 2)
        
        # 合并结果
        combined_scores = {}
        
        # 添加稠密检索结果
        for idx, score, metadata in dense_results:
            combined_scores[idx] = combined_scores.get(idx, 0) + alpha * score
        
        # 添加BM25结果（需要归一化）
        if bm25_results:
            max_bm25_score = max(score for _, score, _ in bm25_results)
            for idx, score, metadata in bm25_results:
                normalized_score = score / max_bm25_score if max_bm25_score > 0 else 0
                combined_scores[idx] = combined_scores.get(idx, 0) + (1 - alpha) * normalized_score
        
        # 排序并返回top-k
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for idx, score in sorted_items:
            if idx < len(self.metadata):
                results.append((idx, score, self.metadata[idx]))
        
        return results

class ChunkLoader:
    """分块数据加载器（用于获取文档内容）"""
    
    def __init__(self, chunks_dir: Path = None):
        """
        初始化分块加载器
        
        Args:
            chunks_dir: 分块数据目录
        """
        if chunks_dir is None:
            from config import CHUNKS_DIR
            chunks_dir = CHUNKS_DIR
        
        self.chunks_dir = chunks_dir
        self.chunk_cache = {}  # 缓存chunk_id到文档内容的映射
    
    def load_chunk_content(self, chunk_id: str) -> Optional[str]:
        """
        根据chunk_id加载文档内容
        
        Args:
            chunk_id: 分块ID（格式: filename_chunkindex）
            
        Returns:
            Optional[str]: 文档内容，如果找不到则返回None
        """
        if chunk_id in self.chunk_cache:
            return self.chunk_cache[chunk_id]
        
        try:
            # 解析chunk_id
            parts = chunk_id.rsplit('_', 1)
            if len(parts) != 2:
                return None
            
            filename, chunk_idx = parts[0], int(parts[1])
            file_path = self.chunks_dir / f"{filename}_chunked.json"
            
            if not file_path.exists():
                return None
            
            # 加载文件
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = data.get('chunks', [])
            if 0 <= chunk_idx < len(chunks):
                content = chunks[chunk_idx].get('text', '')
                self.chunk_cache[chunk_id] = content
                return content
            
        except Exception as e:
            logger.error(f"加载分块内容失败 {chunk_id}: {str(e)}")
        
        return None

def test_dense_retriever():
    """测试稠密检索器"""
    print("=" * 50)
    print("稠密检索器测试")
    print("=" * 50)
    
    # 初始化检索器
    try:
        retriever = DenseRetriever()
        retriever.load_faiss_index()
        print("✅ 检索器初始化成功")
    except Exception as e:
        print(f"❌ 检索器初始化失败: {str(e)}")
        print("请先运行 build_faiss.py 构建向量索引")
        return
    
    # 测试查询
    test_queries = [
        "特斯拉的电池容量是多少？",
        "怎么保养电动汽车？",
        "安全气囊的使用注意事项",
        "自动驾驶功能怎么开启？",
        "车辆无法启动怎么办？"
    ]
    
    # 加载文档内容
    loader = ChunkLoader()
    
    for query in test_queries:
        print(f"\n🔍 查询: {query}")
        print("-" * 50)
        
        results = retriever.search(query, top_k=3)
        
        for i, (idx, score, metadata) in enumerate(results):
            # 获取文档内容
            chunk_id = metadata.get('chunk_id', f'chunk_{idx}')
            content = loader.load_chunk_content(chunk_id)
            
            if content is None:
                content = metadata.get('text', '内容未找到')
            
            source = metadata.get('source', '未知')
            page = metadata.get('page_number', '未知')
            
            print(f"{i+1}. [相似度: {score:.4f}] [{source} 第{page}页]")
            print(f"   内容: {content[:150]}...")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='稠密检索器')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # search命令
    search_parser = subparsers.add_parser('search', help='搜索文档')
    search_parser.add_argument('query', type=str, help='查询文本')
    search_parser.add_argument('--topk', type=int, default=5, help='返回前k个结果')
    search_parser.add_argument('--threshold', type=float, default=0.0, help='相似度阈值')
    
    # batch命令
    batch_parser = subparsers.add_parser('batch', help='批量搜索')
    batch_parser.add_argument('--file', type=str, required=True, help='查询文件路径（每行一个查询）')
    batch_parser.add_argument('--topk', type=int, default=5, help='返回前k个结果')
    batch_parser.add_argument('--output', type=str, default='batch_results.json', help='输出文件路径')
    
    # test命令
    test_parser = subparsers.add_parser('test', help='运行测试')
    
    args = parser.parse_args()
    
    if args.command == 'test':
        test_dense_retriever()
        return
    
    # 初始化检索器
    try:
        retriever = DenseRetriever()
        retriever.load_faiss_index()
        print(f"✅ 检索器初始化成功，包含 {retriever.index.ntotal if retriever.index else 0} 个向量")
    except Exception as e:
        print(f"❌ 检索器初始化失败: {str(e)}")
        print("请先运行 build_faiss.py 构建向量索引")
        exit(1)
    
    if args.command == 'search':
        # 单条查询
        print(f"\n🔍 搜索查询: '{args.query}'")
        
        results = retriever.search(args.query, top_k=args.topk, threshold=args.threshold)
        
        if not results:
            print("❌ 未找到相关文档")
        else:
            print(f"\n📊 找到 {len(results)} 个相关文档:")
            
            # 加载文档内容
            loader = ChunkLoader()
            
            for i, (idx, score, metadata) in enumerate(results):
                # 获取文档内容
                chunk_id = metadata.get('chunk_id', f'chunk_{idx}')
                content = loader.load_chunk_content(chunk_id)
                
                if content is None:
                    content = metadata.get('text', '内容未找到')
                
                source = metadata.get('source', '未知')
                page = metadata.get('page_number', '未知')
                
                print(f"\n{i+1}. [相似度: {score:.4f}]")
                print(f"   来源: {source} (第{page}页)")
                print(f"   内容: {content[:200]}...")
    
    elif args.command == 'batch':
        # 批量查询
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 文件不存在: {args.file}")
            exit(1)
        
        # 读取查询
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        if not queries:
            print("❌ 文件为空")
            exit(1)
        
        print(f"处理 {len(queries)} 个查询...")
        
        # 批量搜索
        all_results = retriever.batch_search(queries, top_k=args.topk)
        
        # 保存结果
        output_data = []
        for query, results in zip(queries, all_results):
            query_results = []
            for idx, score, metadata in results:
                query_results.append({
                    "index": int(idx),
                    "score": float(score),
                    "metadata": metadata
                })
            
            output_data.append({
                "query": query,
                "results": query_results
            })
        
        # 保存到文件
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 批量搜索完成!")
        print(f"结果已保存到: {output_path}")
        
        # 显示统计信息
        total_results = sum(len(r) for r in all_results)
        avg_results = total_results / len(queries) if queries else 0
        print(f"总结果数: {total_results}")
        print(f"平均每个查询: {avg_results:.1f} 个结果")
    
    else:
        # 交互模式
        print("稠密检索器 - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        # 加载文档内容
        loader = ChunkLoader()
        
        while True:
            try:
                query = input("\n请输入查询: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not query:
                    continue
                
                results = retriever.search(query, top_k=5)
                
                if not results:
                    print("❌ 未找到相关文档")
                else:
                    print(f"\n📊 找到 {len(results)} 个相关文档:")
                    for i, (idx, score, metadata) in enumerate(results):
                        # 获取文档内容
                        chunk_id = metadata.get('chunk_id', f'chunk_{idx}')
                        content = loader.load_chunk_content(chunk_id)
                        
                        if content is None:
                            content = metadata.get('text', '内容未找到')
                        
                        source = metadata.get('source', '未知')
                        page = metadata.get('page_number', '未知')
                        
                        print(f"\n{i+1}. [相似度: {score:.4f}]")
                        print(f"   来源: {source} (第{page}页)")
                        print(f"   内容: {content[:150]}...")
                
            except KeyboardInterrupt:
                print("\n\n程序已终止")
                break
            except Exception as e:
                print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()