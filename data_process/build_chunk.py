#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本分块模块 - 将清洗后的文本分割为适合RAG的文本块
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Generator
import logging
from tqdm import tqdm

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import CHUNKS_DIR, RAG_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextChunker:
    """文本分块器"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        """
        初始化分块器
        
        Args:
            chunk_size: 分块大小（字符数）
            chunk_overlap: 重叠大小（字符数）
        """
        self.chunk_size = chunk_size or RAG_CONFIG['chunk_size']
        self.chunk_overlap = chunk_overlap or RAG_CONFIG['chunk_overlap']
        
        # 中文句子分割模式
        self.chinese_sentence_pattern = re.compile(
            r'[。！？；：」）】》」』】〉》〕〗〞〟＂＇＂\']+'
        )
        
        # 英文句子分割模式
        self.english_sentence_pattern = re.compile(
            r'[.!?;:)]+\s*'
        )
        
        # 段落分割模式
        self.paragraph_pattern = re.compile(r'\n\s*\n')
        
        # 车辆相关关键词（用于智能分块）
        self.car_keywords = [
            '概述', '简介', '介绍', '前言',
            '操作', '使用', '驾驶', '行驶',
            '保养', '维护', '维修', '检查',
            '安全', '警告', '注意', '提示',
            '故障', '问题', '排除', '解决',
            '规格', '参数', '技术', '性能',
            '电池', '充电', '能源', '续航',
            '空调', '音响', '娱乐', '导航',
            '座椅', '方向盘', '仪表', '灯光'
        ]
    
    def split_by_sentences(self, text: str) -> List[str]:
        """
        按句子分割文本（支持中英文混合）
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 句子列表
        """
        # 首先按段落分割
        paragraphs = self.paragraph_pattern.split(text)
        sentences = []
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            # 混合使用中英文句子分割
            # 先按中文标点分割
            chinese_splits = self.chinese_sentence_pattern.split(paragraph)
            
            # 对每个中文分割结果再按英文标点分割
            for split in chinese_splits:
                if not split.strip():
                    continue
                
                english_splits = self.english_sentence_pattern.split(split)
                for eng_split in english_splits:
                    if eng_split.strip():
                        sentences.append(eng_split.strip())
        
        return sentences
    
    def create_chunks(self, sentences: List[str]) -> List[Dict]:
        """
        从句子列表创建文本块
        
        Args:
            sentences: 句子列表
            
        Returns:
            List[Dict]: 文本块列表
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # 如果当前块为空，直接添加
            if not current_chunk:
                current_chunk.append(sentence)
                current_length = sentence_length
                continue
            
            # 如果添加当前句子不会超过chunk_size，则添加
            if current_length + sentence_length + 1 <= self.chunk_size:  # +1 for space
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                # 保存当前块
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "length": len(chunk_text),
                        "sentence_count": len(current_chunk)
                    })
                
                # 开始新块，考虑重叠
                if self.chunk_overlap > 0 and len(current_chunk) > 1:
                    # 从当前块的末尾取一些句子作为重叠
                    overlap_sentences = []
                    overlap_length = 0
                    
                    for sent in reversed(current_chunk):
                        sent_len = len(sent)
                        if overlap_length + sent_len + 1 <= self.chunk_overlap:
                            overlap_sentences.insert(0, sent)
                            overlap_length += sent_len + 1
                        else:
                            break
                    
                    current_chunk = overlap_sentences
                    current_length = overlap_length
                else:
                    current_chunk = []
                    current_length = 0
                
                # 添加新句子
                current_chunk.append(sentence)
                current_length = sentence_length
        
        # 添加最后一个块
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                "text": chunk_text,
                "length": len(chunk_text),
                "sentence_count": len(current_chunk)
            })
        
        return chunks
    
    def create_semantic_chunks(self, text: str, metadata: Dict) -> List[Dict]:
        """
        创建语义分块（考虑车辆手册的结构）
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            List[Dict]: 语义分块列表
        """
        # 按标题分割（车辆手册通常有清晰的标题）
        sections = self.split_by_headings(text)
        
        chunks = []
        for section in sections:
            # 对每个部分进行句子分割
            sentences = self.split_by_sentences(section['content'])
            
            # 创建分块
            section_chunks = self.create_chunks(sentences)
            
            # 添加语义信息
            for i, chunk in enumerate(section_chunks):
                chunk_with_metadata = {
                    **chunk,
                    "metadata": {
                        **metadata,
                        "section_title": section.get('title', ''),
                        "section_index": section.get('index', 0),
                        "chunk_index": i,
                        "total_chunks_in_section": len(section_chunks),
                        "chunk_strategy": "semantic"
                    }
                }
                chunks.append(chunk_with_metadata)
        
        return chunks
    
    def split_by_headings(self, text: str) -> List[Dict]:
        """
        按标题分割文本（适用于车辆手册）
        
        Args:
            text: 输入文本
            
        Returns:
            List[Dict]: 标题和内容列表
        """
        # 常见的标题模式
        heading_patterns = [
            r'第[一二三四五六七八九十\d]+章\s+[^\n]+',  # 第X章 标题
            r'第[一二三四五六七八九十\d]+节\s+[^\n]+',  # 第X节 标题
            r'\d+\.\d+\s+[^\n]+',  # 1.1 标题
            r'[一二三四五六七八九十]、\s*[^\n]+',  # 一、标题
            r'\(\d+\)\s+[^\n]+',  # (1) 标题
        ]
        
        sections = []
        current_section = {"title": "概述", "content": "", "index": 0}
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是标题
            is_heading = False
            for pattern in heading_patterns:
                if re.match(pattern, line):
                    is_heading = True
                    break
            
            # 检查是否包含车辆关键词（作为潜在标题）
            if not is_heading:
                for keyword in self.car_keywords:
                    if keyword in line and len(line) < 50:  # 短行且包含关键词
                        is_heading = True
                        break
            
            if is_heading:
                # 保存当前部分
                if current_section["content"]:
                    sections.append(current_section.copy())
                
                # 开始新部分
                current_section = {
                    "title": line,
                    "content": "",
                    "index": len(sections)
                }
            else:
                # 添加到当前部分内容
                if current_section["content"]:
                    current_section["content"] += '\n' + line
                else:
                    current_section["content"] = line
        
        # 添加最后一个部分
        if current_section["content"]:
            sections.append(current_section)
        
        # 如果没有找到标题，则整个文本作为一个部分
        if not sections:
            sections = [{
                "title": "内容",
                "content": text,
                "index": 0
            }]
        
        return sections
    
    def process_document(self, document: Dict) -> List[Dict]:
        """
        处理单个文档
        
        Args:
            document: 文档数据
            
        Returns:
            List[Dict]: 分块后的数据
        """
        source = document.get('source', 'unknown')
        pages = document.get('pages', [])
        
        all_chunks = []
        
        for page in pages:
            text = page.get('text', '')
            metadata = page.get('metadata', {}).copy()
            
            # 添加页面级元数据
            metadata.update({
                "source": source,
                "page_number": page.get('page_number', 0),
                "chunking_method": "semantic_with_overlap"
            })
            
            # 创建语义分块
            chunks = self.create_semantic_chunks(text, metadata)
            all_chunks.extend(chunks)
        
        logger.info(f"文档 {source}: 生成 {len(all_chunks)} 个文本块")
        return all_chunks

def process_all_documents():
    """
    处理所有清洗后的文档
    """
    chunker = TextChunker()
    
    # 获取所有清洗后的JSON文件
    json_files = list(CHUNKS_DIR.glob("*_cleaned.json"))
    
    if not json_files:
        logger.warning(f"在 {CHUNKS_DIR} 中未找到任何清洗后的文件")
        logger.info("请先运行 clean_chunk.py 清洗文档")
        return
    
    logger.info(f"找到 {len(json_files)} 个清洗后的文件")
    
    total_documents = 0
    total_chunks = 0
    
    for json_file in tqdm(json_files, desc="分块处理"):
        try:
            # 读取清洗后的数据
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 处理文档
            chunks = chunker.process_document(data)
            
            if chunks:
                # 保存分块数据
                output_file = CHUNKS_DIR / f"{json_file.stem.replace('_cleaned', '_chunked')}.json"
                
                chunk_data = {
                    "source": data.get('source', 'unknown'),
                    "original_pages": data.get('original_total_pages', 0),
                    "cleaned_pages": data.get('cleaned_total_pages', 0),
                    "total_chunks": len(chunks),
                    "chunk_size": chunker.chunk_size,
                    "chunk_overlap": chunker.chunk_overlap,
                    "chunks": chunks,
                    "chunked_at": str(Path(__file__).parent.parent / "data" / "chunked")
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
                
                total_documents += 1
                total_chunks += len(chunks)
                
                logger.info(f"已分块 {json_file.name}: {len(chunks)} 个文本块")
            else:
                logger.warning(f"文件 {json_file.name} 分块后无有效内容")
                
        except Exception as e:
            logger.error(f"处理文件失败 {json_file}: {str(e)}")
    
    # 统计信息
    logger.info("=" * 50)
    logger.info("文本分块完成!")
    logger.info(f"处理文档数: {total_documents}")
    logger.info(f"总文本块数: {total_chunks}")
    logger.info(f"平均每个文档: {total_chunks/total_documents if total_documents > 0 else 0:.1f} 个文本块")
    logger.info(f"分块大小: {chunker.chunk_size} 字符")
    logger.info(f"重叠大小: {chunker.chunk_overlap} 字符")

def chunk_single_file(input_file: Path, output_file: Path = None):
    """
    分块单个文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（可选）
    """
    chunker = TextChunker()
    
    if not input_file.exists():
        logger.error(f"文件不存在: {input_file}")
        return
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_chunked.json"
    
    try:
        # 读取数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理文档
        chunks = chunker.process_document(data)
        
        if chunks:
            # 保存分块数据
            chunk_data = {
                "source": data.get('source', 'unknown'),
                "original_pages": data.get('original_total_pages', 0),
                "cleaned_pages": data.get('cleaned_total_pages', 0),
                "total_chunks": len(chunks),
                "chunk_size": chunker.chunk_size,
                "chunk_overlap": chunker.chunk_overlap,
                "chunks": chunks,
                "chunked_at": str(Path(__file__).parent.parent / "data" / "chunked")
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"文件分块完成: {input_file.name}")
            logger.info(f"生成文本块数: {len(chunks)}")
            logger.info(f"输出文件: {output_file}")
        else:
            logger.warning(f"文件 {input_file.name} 分块后无有效内容")
            
    except Exception as e:
        logger.error(f"分块文件失败 {input_file}: {str(e)}")

if __name__ == "__main__":
    print("=" * 50)
    print("文本分块工具 - 汽车座舱RAG系统")
    print("=" * 50)
    print(f"输入目录: {CHUNKS_DIR}")
    print(f"分块大小: {RAG_CONFIG['chunk_size']} 字符")
    print(f"重叠大小: {RAG_CONFIG['chunk_overlap']} 字符")
    print("=" * 50)
    
    # 检查输入目录
    if not CHUNKS_DIR.exists():
        print(f"错误: 输入目录不存在: {CHUNKS_DIR}")
        print(f"请先运行 clean_chunk.py 清洗文档")
        exit(1)
    
    # 处理所有文档
    process_all_documents()
    
    print("=" * 50)
    print("分块完成!")
    print("=" * 50)