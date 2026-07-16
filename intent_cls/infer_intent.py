#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图分类推理模块 - 使用训练好的模型进行意图分类
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IntentClassifier:
    """意图分类器"""
    
    def __init__(self, model_path: Optional[Path] = None, device: str = None):
        """
        初始化意图分类器
        
        Args:
            model_path: 模型路径，如果为None则使用默认路径
            device: 运行设备 ("cpu" 或 "cuda")
        """
        if model_path is None:
            model_path = Path(__file__).parent / "models"
        
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self.config = None
        self.intent_classes = []
        self.intent_descriptions = {}
        
        logger.info(f"初始化IntentClassifier，模型路径: {self.model_path}，设备: {self.device}")
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载模型和配置"""
        try:
            # 检查模型是否存在（HuggingFace格式的config.json）
            if not (self.model_path / "config.json").exists():
                logger.warning(f"模型文件不存在: {self.model_path}")
                logger.info("请先运行 train_intent.py 训练模型")
                return
            
            # 加载意图元数据（从intent_config.json，不覆盖HuggingFace的config.json）
            intent_config_file = self.model_path / "intent_config.json"
            if intent_config_file.exists():
                with open(intent_config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            else:
                # 兼容旧格式：尝试从config.json读取（可能已被覆盖）
                config_file = self.model_path / "config.json"
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                # 如果config.json缺少model_type，说明是旧格式被覆盖了
                if "model_type" not in self.config:
                    logger.warning("config.json 似乎被自定义配置覆盖，尝试从checkpoint恢复...")
                    # 尝试从最新的checkpoint加载
                    checkpoint_dirs = list(self.model_path.glob("checkpoint-*"))
                    if checkpoint_dirs:
                        latest_ckpt = max(checkpoint_dirs, key=lambda x: x.stat().st_mtime)
                        logger.info(f"从checkpoint恢复: {latest_ckpt}")
                        self.model_path = latest_ckpt
                        with open(latest_ckpt / "intent_config.json" if (latest_ckpt / "intent_config.json").exists() else latest_ckpt / "config.json", 'r', encoding='utf-8') as f:
                            self.config = json.load(f)
                    else:
                        logger.error("无法恢复模型，请重新训练")
                        return
            
            # 加载意图类别
            self.intent_classes = self.config.get("intent_classes", [])
            self.intent_descriptions = self.config.get("intent_descriptions", {})
            
            # 加载tokenizer和模型
            logger.info(f"正在加载模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
            
            # 移动到设备
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"模型加载成功，意图类别数: {len(self.intent_classes)}")
            
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def predict(self, text: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        预测文本意图
        
        Args:
            text: 输入文本
            top_k: 返回前k个结果
            
        Returns:
            List[Tuple[str, float, str]]: (意图名称, 置信度, 描述) 列表
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        try:
            # 分词
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.config.get("max_length", 128),
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # 获取top-k结果
            probs, indices = torch.topk(probabilities, min(top_k, len(self.intent_classes)))
            probs = probs.cpu().numpy()[0]
            indices = indices.cpu().numpy()[0]
            
            # 构建结果
            results = []
            for prob, idx in zip(probs, indices):
                if idx < len(self.intent_classes):
                    intent_name = self.intent_classes[idx]
                    intent_desc = self.intent_descriptions.get(intent_name, "")
                    results.append((intent_name, float(prob), intent_desc))
            
            return results
            
        except Exception as e:
            logger.error(f"预测失败: {str(e)}")
            return []
    
    def predict_batch(self, texts: List[str], top_k: int = 3) -> List[List[Tuple[str, float, str]]]:
        """
        批量预测文本意图
        
        Args:
            texts: 输入文本列表
            top_k: 返回前k个结果
            
        Returns:
            List[List[Tuple]]: 每个文本的预测结果列表
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")
        
        try:
            # 分词
            inputs = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config.get("max_length", 128),
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 批量推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # 获取每个文本的top-k结果
            all_results = []
            probs, indices = torch.topk(probabilities, min(top_k, len(self.intent_classes)))
            probs = probs.cpu().numpy()
            indices = indices.cpu().numpy()
            
            for i in range(len(texts)):
                text_results = []
                for j in range(top_k):
                    if j < len(probs[i]) and indices[i][j] < len(self.intent_classes):
                        idx = indices[i][j]
                        prob = probs[i][j]
                        intent_name = self.intent_classes[idx]
                        intent_desc = self.intent_descriptions.get(intent_name, "")
                        text_results.append((intent_name, float(prob), intent_desc))
                all_results.append(text_results)
            
            return all_results
            
        except Exception as e:
            logger.error(f"批量预测失败: {str(e)}")
            return [[] for _ in texts]
    
    def get_intent_info(self, intent_name: str) -> Optional[Dict[str, Any]]:
        """
        获取意图的详细信息
        
        Args:
            intent_name: 意图名称
            
        Returns:
            Optional[Dict]: 意图信息，如果不存在则返回None
        """
        if intent_name not in self.intent_classes:
            return None
        
        idx = self.intent_classes.index(intent_name)
        return {
            "id": idx,
            "name": intent_name,
            "description": self.intent_descriptions.get(intent_name, ""),
            "examples": self._get_intent_examples(intent_name)
        }
    
    def _get_intent_examples(self, intent_name: str) -> List[str]:
        """获取意图的示例问题"""
        # 这里可以从训练数据中获取实际示例，这里返回一些通用示例
        examples_map = {
            "车辆信息查询": [
                "这辆车的型号是什么？",
                "生产日期是哪一年？",
                "车辆配置有哪些？"
            ],
            "操作指南": [
                "怎么打开空调？",
                "灯光怎么调节？",
                "座椅记忆怎么设置？"
            ],
            "故障诊断": [
                "车辆启动不了怎么办？",
                "发动机故障灯亮了",
                "刹车有异响"
            ],
            "保养维护": [
                "多久需要保养一次？",
                "机油怎么更换？",
                "轮胎多久更换？"
            ],
            "安全警告": [
                "安全带提示",
                "胎压报警",
                "ABS故障警告"
            ],
            "技术规格": [
                "百公里加速时间",
                "最大扭矩是多少？",
                "油耗是多少？"
            ],
            "电池充电": [
                "怎么充电？",
                "充电时间要多久？",
                "快充功率多少？"
            ],
            "娱乐系统": [
                "音响怎么调节？",
                "导航系统怎么用？",
                "语音助手怎么用？"
            ],
            "驾驶辅助": [
                "自适应巡航怎么用？",
                "自动泊车功能",
                "车道保持辅助"
            ],
            "其他问题": [
                "购车优惠",
                "保险怎么买？",
                "二手车价格"
            ]
        }
        
        return examples_map.get(intent_name, [])
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        分析查询，返回详细信息
        
        Args:
            query: 用户查询
            
        Returns:
            Dict: 分析结果
        """
        # 预测意图
        intent_results = self.predict(query, top_k=3)
        
        if not intent_results:
            return {
                "query": query,
                "primary_intent": None,
                "confidence": 0.0,
                "all_intents": [],
                "suggested_actions": []
            }
        
        # 主要意图
        primary_intent, confidence, description = intent_results[0]
        
        # 建议操作
        suggested_actions = self._get_suggested_actions(primary_intent, query)
        
        return {
            "query": query,
            "primary_intent": {
                "name": primary_intent,
                "confidence": confidence,
                "description": description
            },
            "all_intents": [
                {
                    "name": intent,
                    "confidence": conf,
                    "description": desc
                }
                for intent, conf, desc in intent_results
            ],
            "suggested_actions": suggested_actions
        }
    
    def _get_suggested_actions(self, intent: str, query: str) -> List[str]:
        """根据意图获取建议操作"""
        actions_map = {
            "车辆信息查询": [
                "检索车辆技术规格文档",
                "查找车辆配置信息",
                "提供车辆型号详细信息"
            ],
            "操作指南": [
                "查找相关操作手册",
                "提供步骤指导",
                "显示操作示意图"
            ],
            "故障诊断": [
                "检索故障代码说明",
                "提供故障排除步骤",
                "建议维修方案"
            ],
            "保养维护": [
                "查找保养计划表",
                "提供维护操作指南",
                "显示保养注意事项"
            ],
            "安全警告": [
                "检索安全警告信息",
                "提供安全操作指南",
                "显示紧急处理步骤"
            ],
            "技术规格": [
                "检索技术参数文档",
                "提供性能指标数据",
                "比较不同配置差异"
            ],
            "电池充电": [
                "检索电池使用指南",
                "提供充电操作步骤",
                "显示充电注意事项"
            ],
            "娱乐系统": [
                "检索娱乐系统手册",
                "提供功能操作指南",
                "显示系统设置方法"
            ],
            "驾驶辅助": [
                "检索驾驶辅助说明",
                "提供功能启用步骤",
                "显示使用注意事项"
            ],
            "其他问题": [
                "进行通用信息检索",
                "提供相关文档链接",
                "建议联系客服"
            ]
        }
        
        return actions_map.get(intent, ["进行相关信息检索"])

def test_classifier():
    """测试分类器"""
    print("=" * 50)
    print("意图分类器测试")
    print("=" * 50)
    
    # 初始化分类器
    try:
        classifier = IntentClassifier()
        print("✅ 分类器初始化成功")
    except Exception as e:
        print(f"❌ 分类器初始化失败: {str(e)}")
        print("请先运行 train_intent.py 训练模型")
        return
    
    # 测试查询
    test_queries = [
        "特斯拉Model 3的电池容量是多少？",
        "怎么打开宝马X5的空调？",
        "车辆启动不了怎么办？",
        "多久需要保养一次？",
        "安全气囊注意事项",
        "百公里加速时间",
        "怎么给电动车充电？",
        "音响怎么调节？",
        "自适应巡航怎么用？",
        "这辆车有优惠吗？"
    ]
    
    for query in test_queries:
        print(f"\n📝 查询: {query}")
        print("-" * 40)
        
        results = classifier.predict(query, top_k=2)
        
        for intent, confidence, description in results:
            print(f"  • {intent}: {confidence:.2%} - {description}")
        
        # 详细分析
        analysis = classifier.analyze_query(query)
        if analysis["primary_intent"]:
            print(f"  🎯 主要意图: {analysis['primary_intent']['name']} ({analysis['primary_intent']['confidence']:.2%})")
            print(f"  📋 建议操作: {', '.join(analysis['suggested_actions'][:2])}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='意图分类推理')
    parser.add_argument('--text', type=str, help='要分类的文本')
    parser.add_argument('--file', type=str, help='包含文本的文件路径（每行一个）')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--topk', type=int, default=3, help='返回前k个结果')
    parser.add_argument('--test', action='store_true', help='运行测试')
    
    args = parser.parse_args()
    
    if args.test:
        test_classifier()
        return
    
    # 初始化分类器
    try:
        model_path = Path(args.model) if args.model else None
        classifier = IntentClassifier(model_path)
    except Exception as e:
        print(f"❌ 初始化分类器失败: {str(e)}")
        exit(1)
    
    # 处理文本或文件
    if args.text:
        results = classifier.predict(args.text, top_k=args.topk)
        
        print(f"\n📝 查询: {args.text}")
        print("=" * 50)
        
        for i, (intent, confidence, description) in enumerate(results, 1):
            print(f"{i}. {intent}: {confidence:.2%}")
            print(f"   描述: {description}")
        
        # 详细分析
        analysis = classifier.analyze_query(args.text)
        if analysis["primary_intent"]:
            print(f"\n🎯 主要意图: {analysis['primary_intent']['name']}")
            print(f"📊 置信度: {analysis['primary_intent']['confidence']:.2%}")
            print(f"📋 建议操作:")
            for action in analysis["suggested_actions"]:
                print(f"   • {action}")
    
    elif args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"❌ 文件不存在: {args.file}")
            exit(1)
        
        # 读取文件
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            print("❌ 文件为空")
            exit(1)
        
        # 批量预测
        print(f"处理 {len(texts)} 个文本...")
        all_results = classifier.predict_batch(texts, top_k=args.topk)
        
        for i, (text, results) in enumerate(zip(texts, all_results)):
            print(f"\n{i+1}. {text}")
            print("-" * 40)
            
            for intent, confidence, description in results:
                print(f"   • {intent}: {confidence:.2%}")
    
    else:
        # 交互模式
        print("意图分类器 - 交互模式")
        print("输入 'quit' 或 'exit' 退出")
        print("=" * 50)
        
        classifier = IntentClassifier()
        
        while True:
            try:
                query = input("\n请输入问题: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("再见！")
                    break
                
                if not query:
                    continue
                
                results = classifier.predict(query, top_k=args.topk)
                
                print(f"\n🔍 分析结果:")
                for i, (intent, confidence, description) in enumerate(results, 1):
                    print(f"{i}. {intent}: {confidence:.2%}")
                    print(f"   描述: {description}")
                
            except KeyboardInterrupt:
                print("\n\n程序已终止")
                break
            except Exception as e:
                print(f"错误: {str(e)}")

if __name__ == "__main__":
    main()