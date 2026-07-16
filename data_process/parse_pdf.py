#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF解析模块 - 从车辆手册PDF文件中提取文本
"""

import os
import json
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
import logging
from tqdm import tqdm

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import RAW_DATA_DIR, CHUNKS_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PDFParser:
    """PDF解析器"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.txt', '.md', '.docx']
    
    def parse_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        解析PDF文件，提取文本和元数据
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            List[Dict]: 包含页面文本和元数据的列表
        """
        try:
            logger.info(f"开始解析PDF文件: {pdf_path.name}")
            
            pages_data = []
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                
                for page_num, page in enumerate(tqdm(pdf.pages, desc=f"解析 {pdf_path.name}")):
                    # 提取文本
                    text = page.extract_text()
                    
                    if not text or text.strip() == "":
                        # 尝试使用不同的提取策略
                        text = page.extract_text(x_tolerance=2, y_tolerance=2)
                    
                    if text:
                        # 清理文本
                        text = self._clean_text(text)
                        
                        # 提取页面信息
                        page_data = {
                            "page_number": page_num + 1,
                            "text": text,
                            "metadata": {
                                "source_file": pdf_path.name,
                                "total_pages": total_pages,
                                "page_width": page.width,
                                "page_height": page.height,
                                "rotation": page.rotation
                            }
                        }
                        pages_data.append(page_data)
                    else:
                        logger.warning(f"页面 {page_num + 1} 未提取到文本")
            
            logger.info(f"成功解析 {pdf_path.name}: {len(pages_data)} 页")
            return pages_data
            
        except Exception as e:
            logger.error(f"解析PDF文件失败 {pdf_path}: {str(e)}")
            return []
    
    def parse_txt(self, txt_path: Path) -> List[Dict]:
        """
        解析文本文件
        
        Args:
            txt_path: 文本文件路径
            
        Returns:
            List[Dict]: 包含文本和元数据的列表
        """
        try:
            logger.info(f"开始解析文本文件: {txt_path.name}")
            
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 按段落分割
            paragraphs = content.split('\n\n')
            pages_data = []
            
            for para_idx, paragraph in enumerate(paragraphs):
                if paragraph.strip():
                    page_data = {
                        "page_number": para_idx + 1,
                        "text": self._clean_text(paragraph),
                        "metadata": {
                            "source_file": txt_path.name,
                            "total_paragraphs": len(paragraphs),
                            "paragraph_index": para_idx
                        }
                    }
                    pages_data.append(page_data)
            
            logger.info(f"成功解析 {txt_path.name}: {len(pages_data)} 段落")
            return pages_data
            
        except Exception as e:
            logger.error(f"解析文本文件失败 {txt_path}: {str(e)}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        # 移除多余的空格和换行
        text = ' '.join(text.split())
        
        # 移除特殊字符（保留中文、英文、数字和标点）
        import re
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s，。！？；："\'、（）《》【】]', '', text)
        
        # 标准化标点
        text = text.replace('，', ', ').replace('。', '. ').replace('！', '! ').replace('？', '? ')
        
        return text.strip()
    
    def save_to_json(self, pages_data: List[Dict], output_path: Path):
        """
        将解析结果保存为JSON文件
        
        Args:
            pages_data: 页面数据列表
            output_path: 输出文件路径
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "source": pages_data[0]["metadata"]["source_file"] if pages_data else "unknown",
                    "total_pages": len(pages_data),
                    "pages": pages_data,
                    "parsed_at": str(Path(__file__).parent.parent / "data" / "parsed")
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存JSON文件失败 {output_path}: {str(e)}")

def process_all_documents():
    """
    处理raw目录中的所有文档
    """
    parser = PDFParser()
    
    # 确保输出目录存在
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 获取所有支持的文档文件
    document_files = []
    for ext in parser.supported_extensions:
        document_files.extend(list(RAW_DATA_DIR.glob(f"*{ext}")))
    
    if not document_files:
        logger.warning(f"在 {RAW_DATA_DIR} 中未找到任何文档文件")
        logger.info(f"支持的格式: {', '.join(parser.supported_extensions)}")
        return
    
    logger.info(f"找到 {len(document_files)} 个文档文件")
    
    total_pages = 0
    for doc_file in document_files:
        logger.info(f"处理文件: {doc_file.name}")
        
        # 根据文件类型选择解析方法
        if doc_file.suffix.lower() == '.pdf':
            pages_data = parser.parse_pdf(doc_file)
        elif doc_file.suffix.lower() in ['.txt', '.md']:
            pages_data = parser.parse_txt(doc_file)
        elif doc_file.suffix.lower() == '.docx':
            # TODO: 添加docx解析支持
            logger.warning(f"暂不支持 {doc_file.suffix} 格式: {doc_file.name}")
            continue
        else:
            logger.warning(f"不支持的文件格式: {doc_file.suffix}")
            continue
        
        if pages_data:
            # 保存解析结果
            output_file = CHUNKS_DIR / f"{doc_file.stem}_parsed.json"
            parser.save_to_json(pages_data, output_file)
            total_pages += len(pages_data)
    
    logger.info(f"文档处理完成! 共处理 {len(document_files)} 个文件，{total_pages} 页/段落")

if __name__ == "__main__":
    print("=" * 50)
    print("PDF解析工具 - 汽车座舱RAG系统")
    print("=" * 50)
    print(f"输入目录: {RAW_DATA_DIR}")
    print(f"输出目录: {CHUNKS_DIR}")
    print("=" * 50)
    
    # 检查输入目录
    if not RAW_DATA_DIR.exists():
        print(f"错误: 输入目录不存在: {RAW_DATA_DIR}")
        print(f"请将车辆手册文件放入: {RAW_DATA_DIR}")
        exit(1)
    
    # 处理所有文档
    process_all_documents()
    
    print("=" * 50)
    print("处理完成!")
    print("=" * 50)