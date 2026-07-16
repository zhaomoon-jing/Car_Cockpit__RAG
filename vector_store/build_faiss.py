#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAISS向量索引构建模块 - 构建和存储文本向量索引
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
import faiss

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNKS_DIR, FAISS_INDEX_DIR, MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FAISSIndexBuilder:
    """FAISS索引构建器"""
    
    def __init__(self, model_name: str = None, embedding_dim: int = None):
        """
        初始化索引构建器
        
        Args:
            model_name: 嵌入模型名称
            embedding_dim: 嵌入维度
        """
        self.model_name = model_name or MODEL_CONFIG['embedding_model']
        self.embedding_dim = embedding_dim or MODEL_CONFIG['embedding_dim']
        
        # 延迟加载模型
        self.model = None
        self.tokenizer = None
        
        # FAISS索引
        self.index = None
        self.metadata = []
        
        logger.info(f"使用模型: {self.model_name}")
        logger.info(f"嵌入维度: {self.embedding_dim}")
    
    def load_model(self):
        """加载嵌入模型"""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"正在加载模型: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # 测试模型
            test_embedding = self.model.encode(["测试文本"])
            actual_dim = test_embedding.shape[1]
            
            if actual_dim != self.embedding_dim:
                logger.warning(f"模型维度 ({actual_dim}) 与配置维度 ({self.embedding_dim}) 不匹配，使用实际维度")
                self.embedding_dim = actual_dim
            
            logger.info(f"模型加载成功，实际维度: {self.embedding_dim}")
            
        except ImportError:
            logger.error("请安装 sentence-transformers: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        编码文本列表为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            np.ndarray: 向量矩阵 (n_samples, embedding_dim)
        """
        if self.model is None:
            self.load_model()
        
        logger.info(f"正在编码 {len(texts)} 个文本...")
        
        # 批量编码
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="编码文本"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch, show_progress_bar=False)
            embeddings.append(batch_embeddings)
        
        # 合并所有批次的嵌入
        all_embeddings = np.vstack(embeddings)
        
        # 归一化（对余弦相似度有益）
        faiss.normalize_L2(all_embeddings)
        
        logger.info(f"编码完成，形状: {all_embeddings.shape}")
        return all_embeddings
    
    def build_index(self, embeddings: np.ndarray, metadata: List[Dict]) -> faiss.Index:
        """
        构建FAISS索引
        
        Args:
            embeddings: 向量矩阵
            metadata: 元数据列表
            
        Returns:
            faiss.Index: FAISS索引
        """
        n_samples, dim = embeddings.shape
        
        logger.info(f"构建FAISS索引，样本数: {n_samples}, 维度: {dim}")
        
        # 使用IVF索引（适合大规模数据）
        nlist = min(100, int(np.sqrt(n_samples)))  # 聚类中心数
        
        quantizer = faiss.IndexFlatIP(dim)  # 内积（余弦相似度）
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # 训练索引
        if n_samples >= nlist * 39:  # FAISS推荐的最小训练样本数
            logger.info("训练FAISS索引...")
            index.train(embeddings)
        else:
            logger.warning(f"样本数 ({n_samples}) 不足，使用Flat索引")
            index = faiss.IndexFlatIP(dim)
        
        # 添加向量到索引
        logger.info("添加向量到索引...")
        index.add(embeddings)
        
        # 保存元数据
        self.metadata = metadata
        
        logger.info(f"索引构建完成，包含 {index.ntotal} 个向量")
        return index
    
    def save_index(self, index: faiss.Index, output_dir: Path):
        """
        保存索引和元数据
        
        Args:
            index: FAISS索引
            output_dir: 输出目录
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存FAISS索引
        index_file = output_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_file))
        logger.info(f"FAISS索引已保存到: {index_file}")
        
        # 保存元数据
        metadata_file = output_dir / "metadata.pkl"
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"元数据已保存到: {metadata_file}")
        
        # 保存配置信息
        config_file = output_dir / "config.json"
        config = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "total_vectors": index.ntotal,
            "index_type": type(index).__name__,
            "build_time": str(Path(__file__).parent.parent / "vector_store" / "built")
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        logger.info(f"配置已保存到: {config_file}")
    
    def load_chunked_data(self) -> Tuple[List[str], List[Dict]]:
        """
        加载分块后的数据
        
        Returns:
            Tuple[List[str], List[Dict]]: 文本列表和元数据列表
        """
        # 获取所有分块后的JSON文件
        json_files = list(CHUNKS_DIR.glob("*_chunked.json"))
        
        if not json_files:
            logger.error(f"在 {CHUNKS_DIR} 中未找到任何分块文件")
            logger.info("请先运行 build_chunk.py 分块文档")
            return [], []
        
        logger.info(f"找到 {len(json_files)} 个分块文件")
        
        all_texts = []
        all_metadata = []
        
        for json_file in tqdm(json_files, desc="加载分块数据"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                chunks = data.get('chunks', [])
                
                for chunk in chunks:
                    text = chunk.get('text', '')
                    metadata = chunk.get('metadata', {}).copy()
                    
                    # 添加分块信息
                    metadata.update({
                        "source_file": json_file.name,
                        "chunk_id": f"{json_file.stem}_{len(all_texts)}",
                        "text_length": len(text)
                    })
                    
                    all_texts.append(text)
                    all_metadata.append(metadata)
                    
            except Exception as e:
                logger.error(f"加载文件失败 {json_file}: {str(e)}")
        
        logger.info(f"加载完成: {len(all_texts)} 个文本块")
        return all_texts, all_metadata
    
    def build_from_chunks(self):
        """从分块数据构建索引"""
        # 加载数据
        texts, metadata = self.load_chunked_data()
        
        if not texts:
            logger.error("没有可用的文本数据")
            return
        
        # 编码文本
        embeddings = self.encode_texts(texts)
        
        # 构建索引
        index = self.build_index(embeddings, metadata)
        
        # 保存索引
        self.save_index(index, FAISS_INDEX_DIR)
        
        # 打印统计信息
        self.print_statistics(texts, metadata)
    
    def print_statistics(self, texts: List[str], metadata: List[Dict]):
        """打印统计信息"""
        logger.info("=" * 50)
        logger.info("索引构建统计信息")
        logger.info("=" * 50)
        
        # 文本长度统计
        text_lengths = [len(text) for text in texts]
        logger.info(f"文本块总数: {len(texts)}")
        logger.info(f"平均文本长度: {np.mean(text_lengths):.1f} 字符")
        logger.info(f"最小文本长度: {np.min(text_lengths)} 字符")
        logger.info(f"最大文本长度: {np.max(text_lengths)} 字符")
        
        # 来源文件统计
        source_files = {}
        for meta in metadata:
            source = meta.get('source', 'unknown')
            source_files[source] = source_files.get(source, 0) + 1
        
        logger.info(f"来源文件数: {len(source_files)}")
        for source, count in list(source_files.items())[:5]:  # 显示前5个
            logger.info(f"  {source}: {count} 个文本块")
        
        if len(source_files) > 5:
            logger.info(f"  ... 和 {len(source_files) - 5} 个其他文件")
        
        # 内存使用估计
        vector_memory = len(texts) * self.embedding_dim * 4 / 1024 / 1024  # MB
        logger.info(f"向量内存占用: {vector_memory:.2f} MB")
        
        logger.info("=" * 50)

def main():
    """主函数"""
    print("=" * 50)
    print("FAISS向量索引构建工具 - 汽车座舱RAG系统")
    print("=" * 50)
    print(f"输入目录: {CHUNKS_DIR}")
    print(f"输出目录: {FAISS_INDEX_DIR}")
    print(f"嵌入模型: {MODEL_CONFIG['embedding_model']}")
    print(f"嵌入维度: {MODEL_CONFIG['embedding_dim']}")
    print("=" * 50)
    
    # 检查输入目录
    if not CHUNKS_DIR.exists():
        print(f"错误: 输入目录不存在: {CHUNKS_DIR}")
        print(f"请先运行 build_chunk.py 分块文档")
        exit(1)
    
    # 创建输出目录
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # 构建索引
        builder = FAISSIndexBuilder()
        builder.build_from_chunks()
        
        print("=" * 50)
        print("✅ 索引构建完成!")
        print(f"索引文件保存在: {FAISS_INDEX_DIR}")
        print("=" * 50)
        
        # 显示使用说明
        print("\n📖 使用说明:")
        print("1. 索引文件:")
        print(f"   • faiss_index.bin - FAISS索引文件")
        print(f"   • metadata.pkl - 文本块元数据")
        print(f"   • config.json - 索引配置")
        print("\n2. 在代码中使用索引:")
        print("   ```python")
        print("   import faiss")
        print("   import pickle")
        print("   ")
        print("   # 加载索引")
        print(f'   index = faiss.read_index("{FAISS_INDEX_DIR}/faiss_index.bin")')
        print(f'   with open("{FAISS_INDEX_DIR}/metadata.pkl", "rb") as f:')
        print("       metadata = pickle.load(f)")
        print("   ```")
        
    except Exception as e:
        print(f"❌ 索引构建失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()