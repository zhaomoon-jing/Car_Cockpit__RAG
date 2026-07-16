#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BM25检索器 - 基于关键词的稀疏检索
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from rank_bm25 import BM25Okapi
from tqdm import tqdm
import jieba

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNKS_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ChineseTokenizer:
    """中文分词器"""
    
    def __init__(self, use_jieba: bool = True):
        """
        初始化中文分词器
        
        Args:
            use_jieba: 是否使用jieba分词
        """
        self.use_jieba = use_jieba
        
        if use_jieba:
            # 初始化jieba，添加车辆相关词汇
            self._init_jieba()
    
    def _init_jieba(self):
        """初始化jieba分词器"""
        # 添加车辆相关词汇到词典
        car_terms = [
            "特斯拉", "宝马", "奔驰", "奥迪", "丰田", "本田", "大众", "比亚迪", "蔚来", "理想",
            "电动车", "新能源汽车", "混合动力", "燃油车", "SUV", "轿车", "MPV",
            "电池", "电机", "电控", "续航", "充电", "快充", "慢充", "换电",
            "自动驾驶", "辅助驾驶", "自适应巡航", "车道保持", "自动泊车",
            "智能座舱", "车载系统", "中控屏", "仪表盘", "HUD",
            "安全气囊", "ABS", "ESP", "胎压监测", "盲点监测",
            "保养", "维修", "故障", "诊断", "召回",
            "用户手册", "维修手册", "技术手册", "保养手册"
        ]
        
        for term in car_terms:
            jieba.add_word(term)
    
    def tokenize(self, text: str) -> List[str]:
        """
        中文分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        if not text or not isinstance(text, str):
            return []
        
        # 清理文本
        text = self._clean_text(text)
        
        if self.use_jieba:
            # 使用jieba分词
            tokens = jieba.lcut(text)
        else:
            # 简单空格分割（英文）
            tokens = text.split()
        
        # 过滤停用词和短词
        tokens = [token for token in tokens if self._is_valid_token(token)]
        
        return tokens
    
    def _clean_text(self, text: str) -> str:
        """清理文本"""
        import re
        
        # 移除特殊字符
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s]', ' ', text)
        
        # 移除多余空格
        text = ' '.join(text.split())
        
        return text.strip()
    
    def _is_valid_token(self, token: str) -> bool:
        """判断是否是有效token"""
        if not token or len(token) < 2:
            return False
        
        # 过滤纯数字
        if token.isdigit():
            return False
        
        # 过滤单个字符（除非是中文）
        if len(token) == 1 and not '\u4e00-\u9fa5' in token:
            return False
        
        return True

class BM25Retriever:
    """BM25检索器"""
    
    def __init__(self, tokenizer: Optional[ChineseTokenizer] = None):
        """
        初始化BM25检索器
        
        Args:
            tokenizer: 分词器，如果为None则使用默认中文分词器
        """
        self.tokenizer = tokenizer or ChineseTokenizer()
        self.bm25 = None
        self.documents = []
        self.metadata = []
        
        logger.info("初始化BM25检索器")
    
    def build_index(self, documents: List[str], metadata: Optional[List[Dict]] = None):
        """
        构建BM25索引
        
        Args:
            documents: 文档列表
            metadata: 元数据列表
        """
        if not documents:
            raise ValueError("文档列表不能为空")
        
        logger.info(f"构建BM25索引，文档数: {len(documents)}")
        
        # 分词
        tokenized_docs = []
        for doc in tqdm(documents, desc="文档分词"):
            tokens = self.tokenizer.tokenize(doc)
            tokenized_docs.append(tokens)
        
        # 构建BM25索引
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # 保存文档和元数据
        self.documents = documents
        self.metadata = metadata if metadata else [{} for _ in documents]
        
        logger.info(f"BM25索引构建完成，包含 {len(documents)} 个文档")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        搜索相似文档
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[int, float, Dict]]: (文档索引, 分数, 元数据) 列表
        """
        if self.bm25 is None:
            raise RuntimeError("请先调用 build_index() 构建索引")
        
        # 查询分词
        query_tokens = self.tokenizer.tokenize(query)
        
        if not query_tokens:
            logger.warning(f"查询分词结果为空: {query}")
            return []
        
        # 计算BM25分数
        scores = self.bm25.get_scores(query_tokens)
        
        # 获取top-k结果
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # 只返回分数大于0的结果
                results.append((idx, float(scores[idx]), self.metadata[idx]))
        
        return results
    
    def search_with_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似文档并返回文档内容
        
        Args:
            query: 查询文本
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[str, float, Dict]]: (文档内容, 分数, 元数据) 列表
        """
        results = self.search(query, top_k)
        
        return [
            (self.documents[idx], score, metadata)
            for idx, score, metadata in results
        ]
    
    def save_index(self, output_dir: Path):
        """
        保存索引
        
        Args:
            output_dir: 输出目录
        """
        if self.bm25 is None:
            raise RuntimeError("索引未构建")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        index_file = output_dir / "bm25_index.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'documents': self.documents,
                'metadata': self.metadata
            }, f)
        
        logger.info(f"BM25索引已保存到: {index_file}")
        
        # 保存配置
        config_file = output_dir / "bm25_config.json"
        config = {
            "total_documents": len(self.documents),
            "tokenizer_type": "jieba" if self.tokenizer.use_jieba else "simple",
            "save_time": str(Path(__file__).parent.parent / "retriever" / "saved")
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"配置已保存到: {config_file}")
    
    def load_index(self, index_path: Path):
        """
        加载索引
        
        Args:
            index_path: 索引文件路径
        """
        if not index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {index_path}")
        
        logger.info(f"加载BM25索引: {index_path}")
        
        with open(index_path, 'rb') as f:
            data = pickle.load(f)
        
        self.bm25 = data['bm25']
        self.documents = data['documents']
        self.metadata = data['metadata']
        
        logger.info(f"BM25索引加载完成，包含 {len(self.documents)} 个文档")

class ChunkLoader:
    """分块数据加载器"""
    
    def __init__(self, chunks_dir: Path = CHUNKS_DIR):
        """
        初始化分块加载器
        
        Args:
            chunks_dir: 分块数据目录
        """
        self.chunks_dir = chunks_dir
    
    def load_all_chunks(self) -> Tuple[List[str], List[Dict]]:
        """
        加载所有分块数据
        
        Returns:
            Tuple[List[str], List[Dict]]: (文本列表, 元数据列表)
        """
        # 获取所有分块后的JSON文件
        json_files = list(self.chunks_dir.glob("*_chunked.json"))
        
        if not json_files:
            logger.warning(f"在 {self.chunks_dir} 中未找到任何分块文件")
            return [], []
        
        logger.info(f"找到 {len(json_files)} 个分块文件")
        
        all_texts = []
        all_metadata = []
        
        for json_file in tqdm(json_files, desc="加载分块数据"):
            try:
                texts, metadata = self._load_chunk_file(json_file)
                all_texts.extend(texts)
                all_metadata.extend(metadata)
                
            except Exception as e:
                logger.error(f"加载文件失败 {json_file}: {str(e)}")
        
        logger.info(f"加载完成: {len(all_texts)} 个文本块")
        return all_texts, all_metadata
    
    def _load_chunk_file(self, file_path: Path) -> Tuple[List[str], List[Dict]]:
        """
        加载单个分块文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            Tuple[List[str], List[Dict]]: (文本列表, 元数据列表)
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        chunks = data.get('chunks', [])
        
        texts = []
        metadata_list = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            metadata = chunk.get('metadata', {}).copy()
            
            # 添加文件信息
            metadata.update({
                "source_file": file_path.name,
                "chunk_id": f"{file_path.stem}_{len(texts)}"
            })
            
            texts.append(text)
            metadata_list.append(metadata)
        
        return texts, metadata_list

def build_bm25_index():
    """构建BM25索引"""
    print("=" * 50)
    print("BM25索引构建工具 - 汽车座舱RAG系统")
    print("=" * 50)
    
    # 加载分块数据
    print("📂 加载分块数据...")
    loader = ChunkLoader()
    texts, metadata = loader.load_all_chunks()
    
    if not texts:
        print("❌ 没有可用的文本数据")
        print("请先运行 build_chunk.py 分块文档")
        exit(1)
    
    print(f"✅ 加载完成: {len(texts)} 个文本块")
    
    # 构建BM25索引
    print("\n🔨 构建BM25索引...")
    retriever = BM25Retriever()
    retriever.build_index(texts, metadata)
    
    # 保存索引
    output_dir = Path(__file__).parent / "bm25_index"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    retriever.save_index(output_dir)
    
    print(f"\n✅ BM25索引构建完成!")
    print(f"索引文件保存在: {output_dir}")
    
    # 测试检索
    print("\n🧪 测试检索...")
    test_queries = [
        "特斯拉电池容量",
        "怎么保养车辆",
        "安全气囊注意事项",
        "自动驾驶功能"
    ]
    
    for query in test_queries:
        results = retriever.search_with_documents(query, top_k=2)
        print(f"\n查询: '{query}'")
        for i, (doc, score, meta) in enumerate(results):
            source = meta.get('source', '未知')
            print(f"  {i+1}. [分数: {score:.4f}] [{source}]")
            print(f"     内容: {doc[:100]}...")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='BM25检索器')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # build命令
    build_parser = subparsers.add_parser('build', help='构建BM25索引')
    
    # search命令
    search_parser = subparsers.add_parser('search', help='搜索文档')
    search_parser.add_argument('query', type=str, help='查询文本')
    search_parser.add_argument('--index', type=str, default='bm25_index/bm25_index.pkl', help='索引文件路径')
    search_parser.add_argument('--topk', type=int, default=5, help='返回前k个结果')
    
    args = parser.parse_args()
    
    if args.command == 'build':
        build_bm25_index()
    
    elif args.command == 'search':
        # 加载索引
        index_path = Path(args.index)
        if not index_path.exists():
            print(f"❌ 索引文件不存在: {index_path}")
            print("请先运行 'python bm25_retriever.py build' 构建索引")
            exit(1)
        
        # 初始化检索器
        retriever = BM25Retriever()
        retriever.load_index(index_path)
        
        # 搜索
        print(f"\n🔍 搜索查询: '{args.query}'")
        results = retriever.search_with_documents(args.query, top_k=args.topk)
        
        if not results:
            print("❌ 未找到相关文档")
        else:
            print(f"\n📊 找到 {len(results)} 个相关文档:")
            for i, (doc, score, meta) in enumerate(results):
                source = meta.get('source', '未知')
                page = meta.get('page_number', '未知')
                print(f"\n{i+1}. [相关度: {score:.4f}]")
                print(f"   来源: {source} (第{page}页)")
                print(f"   内容: {doc[:200]}...")
    
    else:
        # 交互模式
        print("BM25检索器 - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        # 检查索引是否存在
        index_path = Path(__file__).parent / "bm25_index" / "bm25_index.pkl"
        
        if not index_path.exists():
            print("❌ 未找到BM25索引")
            print("请先运行 'python bm25_retriever.py build' 构建索引")
            return
        
        # 加载索引
        retriever = BM25Retriever()
        retriever.load_index(index_path)
        print(f"✅ 索引加载完成，包含 {len(retriever.documents)} 个文档")
        
        while True:
            try:
                query = input("\n请输入查询: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not query:
                    continue
                
                results = retriever.search_with_documents(query, top_k=5)
                
                if not results:
                    print("❌ 未找到相关文档")
                else:
                    print(f"\n📊 找到 {len(results)} 个相关文档:")
                    for i, (doc, score, meta) in enumerate(results):
                        source = meta.get('source', '未知')
                        page = meta.get('page_number', '未知')
                        print(f"\n{i+1}. [相关度: {score:.4f}]")
                        print(f"   来源: {source} (第{page}页)")
                        print(f"   内容: {doc[:150]}...")
                
            except KeyboardInterrupt:
                print("\n\n程序已终止")
                break
            except Exception as e:
                print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()