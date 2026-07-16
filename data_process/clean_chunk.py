#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本清洗模块 - 清理和预处理文档分块
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm

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

class TextCleaner:
    """文本清洗器"""
    
    def __init__(self):
        # 定义需要清理的模式
        self.patterns = {
            # 页码标记
            'page_numbers': re.compile(r'第\s*\d+\s*页'),
            'page_of': re.compile(r'Page\s*\d+\s*of\s*\d+'),
            
            # 页眉页脚
            'header_footer': re.compile(r'^[A-Za-z0-9\s\.\-]+$', re.MULTILINE),
            
            # 网址和邮箱
            'urls': re.compile(r'https?://\S+|www\.\S+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # 特殊字符
            'special_chars': re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'),
            
            # 多余空格
            'multiple_spaces': re.compile(r'\s+'),
            
            # 空白行
            'blank_lines': re.compile(r'\n\s*\n'),
        }
        
        # 车辆相关停用词（可根据需要扩展）
        self.stopwords = {
            '版权所有', '保留所有权利', 'All rights reserved',
            '保密', '机密', 'Confidential',
            '版本', 'Version', 'V',
            '日期', 'Date',
            '修订', 'Revision',
            '页码', 'Page',
            '目录', 'Table of Contents',
            '索引', 'Index',
            '附录', 'Appendix',
            '参考文献', 'References',
        }
    
    def clean_text(self, text: str) -> str:
        """
        清理单段文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if not text or not text.strip():
            return ""
        
        # 移除特殊字符
        text = self.patterns['special_chars'].sub(' ', text)
        
        # 移除网址和邮箱
        text = self.patterns['urls'].sub('', text)
        text = self.patterns['emails'].sub('', text)
        
        # 移除页码标记
        text = self.patterns['page_numbers'].sub('', text)
        text = self.patterns['page_of'].sub('', text)
        
        # 处理页眉页脚（单行且只有字母数字和标点）
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and not self.patterns['header_footer'].fullmatch(line):
                # 移除停用词开头的行
                if not any(line.startswith(sw) for sw in self.stopwords):
                    cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        # 标准化空格
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        # 移除空白行
        text = self.patterns['blank_lines'].sub('\n', text)
        
        # 移除首尾空白
        text = text.strip()
        
        return text
    
    def clean_chunk(self, chunk: Dict) -> Dict:
        """
        清理单个文本块
        
        Args:
            chunk: 原始文本块
            
        Returns:
            Dict: 清理后的文本块
        """
        cleaned_chunk = chunk.copy()
        
        # 清理文本内容
        if 'text' in chunk:
            cleaned_chunk['text'] = self.clean_text(chunk['text'])
        
        # 清理元数据中的无用信息
        if 'metadata' in chunk:
            metadata = chunk['metadata'].copy()
            
            # 移除空值的元数据字段
            cleaned_metadata = {k: v for k, v in metadata.items() if v is not None and v != ''}
            
            # 标准化字段名
            if 'source_file' in cleaned_metadata:
                cleaned_metadata['source'] = cleaned_metadata.pop('source_file')
            
            cleaned_chunk['metadata'] = cleaned_metadata
        
        # 添加清理标记
        cleaned_chunk['cleaned'] = True
        cleaned_chunk['original_length'] = len(chunk.get('text', ''))
        cleaned_chunk['cleaned_length'] = len(cleaned_chunk.get('text', ''))
        
        return cleaned_chunk
    
    def should_keep_chunk(self, chunk: Dict) -> bool:
        """
        判断是否应该保留该文本块
        
        Args:
            chunk: 文本块
            
        Returns:
            bool: 是否保留
        """
        text = chunk.get('text', '')
        
        # 检查文本是否为空
        if not text or len(text.strip()) < 10:  # 至少10个字符
            return False
        
        # 检查是否包含有效内容（中文或英文单词）
        has_chinese = bool(re.search(r'[\u4e00-\u9fa5]', text))
        has_english = bool(re.search(r'\b[a-zA-Z]{3,}\b', text))
        
        if not (has_chinese or has_english):
            return False
        
        # 检查是否主要是标点或数字
        char_count = len(text)
        punct_count = len(re.findall(r'[^\w\s\u4e00-\u9fa5]', text))
        digit_count = len(re.findall(r'\d', text))
        
        if (punct_count + digit_count) / char_count > 0.7:  # 超过70%是标点或数字
            return False
        
        return True

def process_all_chunks():
    """
    处理所有已解析的文档分块
    """
    cleaner = TextCleaner()
    
    # 获取所有解析后的JSON文件
    json_files = list(CHUNKS_DIR.glob("*_parsed.json"))
    
    if not json_files:
        logger.warning(f"在 {CHUNKS_DIR} 中未找到任何解析文件")
        logger.info("请先运行 parse_pdf.py 解析文档")
        return
    
    logger.info(f"找到 {len(json_files)} 个解析文件")
    
    total_original_chunks = 0
    total_cleaned_chunks = 0
    
    for json_file in tqdm(json_files, desc="清洗文档"):
        try:
            # 读取原始数据
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 清理每个页面/段落
            cleaned_pages = []
            original_pages = data.get('pages', [])
            total_original_chunks += len(original_pages)
            
            for page in original_pages:
                cleaned_page = cleaner.clean_chunk(page)
                
                # 检查是否应该保留
                if cleaner.should_keep_chunk(cleaned_page):
                    cleaned_pages.append(cleaned_page)
                    total_cleaned_chunks += 1
            
            # 保存清理后的数据
            if cleaned_pages:
                output_file = CHUNKS_DIR / f"{json_file.stem.replace('_parsed', '_cleaned')}.json"
                
                cleaned_data = {
                    "source": data.get('source', 'unknown'),
                    "original_total_pages": len(original_pages),
                    "cleaned_total_pages": len(cleaned_pages),
                    "pages": cleaned_pages,
                    "cleaned_at": str(Path(__file__).parent.parent / "data" / "cleaned")
                }
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
                
                logger.info(f"已清理 {json_file.name}: {len(original_pages)} -> {len(cleaned_pages)} 页")
            else:
                logger.warning(f"文件 {json_file.name} 清理后无有效内容")
                
        except Exception as e:
            logger.error(f"处理文件失败 {json_file}: {str(e)}")
    
    # 统计信息
    logger.info("=" * 50)
    logger.info("文本清洗完成!")
    logger.info(f"原始分块数: {total_original_chunks}")
    logger.info(f"清理后分块数: {total_cleaned_chunks}")
    logger.info(f"过滤分块数: {total_original_chunks - total_cleaned_chunks}")
    logger.info(f"保留率: {(total_cleaned_chunks / total_original_chunks * 100):.1f}%")

def clean_single_file(input_file: Path, output_file: Path = None):
    """
    清理单个文件
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径（可选）
    """
    cleaner = TextCleaner()
    
    if not input_file.exists():
        logger.error(f"文件不存在: {input_file}")
        return
    
    if output_file is None:
        output_file = input_file.parent / f"{input_file.stem}_cleaned.json"
    
    try:
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 清理数据
        cleaned_pages = []
        original_pages = data.get('pages', [])
        
        for page in original_pages:
            cleaned_page = cleaner.clean_chunk(page)
            if cleaner.should_keep_chunk(cleaned_page):
                cleaned_pages.append(cleaned_page)
        
        # 保存清理后的数据
        cleaned_data = {
            "source": data.get('source', 'unknown'),
            "original_total_pages": len(original_pages),
            "cleaned_total_pages": len(cleaned_pages),
            "pages": cleaned_pages,
            "cleaned_at": str(Path(__file__).parent.parent / "data" / "cleaned")
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"文件清理完成: {input_file.name}")
        logger.info(f"原始: {len(original_pages)} 页, 清理后: {len(cleaned_pages)} 页")
        logger.info(f"输出文件: {output_file}")
        
    except Exception as e:
        logger.error(f"清理文件失败 {input_file}: {str(e)}")

if __name__ == "__main__":
    print("=" * 50)
    print("文本清洗工具 - 汽车座舱RAG系统")
    print("=" * 50)
    print(f"输入目录: {CHUNKS_DIR}")
    print("=" * 50)
    
    # 检查输入目录
    if not CHUNKS_DIR.exists():
        print(f"错误: 输入目录不存在: {CHUNKS_DIR}")
        print(f"请先运行 parse_pdf.py 解析文档")
        exit(1)
    
    # 处理所有分块
    process_all_chunks()
    
    print("=" * 50)
    print("清洗完成!")
    print("=" * 50)