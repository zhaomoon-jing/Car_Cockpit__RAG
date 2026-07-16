#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
意图分类模型训练模块 - 训练车辆相关意图分类模型
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import logging
from tqdm import tqdm
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
import evaluate

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class IntentConfig:
    """意图分类配置"""
    model_name: str = MODEL_CONFIG['intent_model']
    num_labels: int = 10  # 默认10个意图类别
    max_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    output_dir: Path = Path(__file__).parent / "models"
    
    # 车辆相关意图类别
    INTENT_CLASSES = [
        "车辆信息查询",      # 0: 查询车辆基本信息
        "操作指南",         # 1: 如何操作车辆功能
        "故障诊断",         # 2: 车辆故障相关问题
        "保养维护",         # 3: 保养和维护问题
        "安全警告",         # 4: 安全相关警告和提示
        "技术规格",         # 5: 技术参数和规格
        "电池充电",         # 6: 电池和充电问题
        "娱乐系统",         # 7: 娱乐和信息系统
        "驾驶辅助",         # 8: 驾驶辅助功能
        "其他问题"          # 9: 其他类型问题
    ]
    
    INTENT_DESCRIPTIONS = {
        "车辆信息查询": "查询车辆型号、配置、生产日期等基本信息",
        "操作指南": "如何操作车辆的各种功能，如空调、灯光、座椅等",
        "故障诊断": "车辆出现故障时的诊断和解决方法",
        "保养维护": "车辆的保养计划、维护方法和注意事项",
        "安全警告": "安全相关的警告、提示和注意事项",
        "技术规格": "车辆的技术参数、性能指标和规格说明",
        "电池充电": "电池使用、充电方法和续航问题",
        "娱乐系统": "音响、导航、娱乐系统的使用",
        "驾驶辅助": "自动驾驶、巡航控制等驾驶辅助功能",
        "其他问题": "不属于以上类别的其他问题"
    }

class IntentDataset:
    """意图分类数据集"""
    
    def __init__(self, config: IntentConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def load_dataset(self, data_dir: Path) -> DatasetDict:
        """
        加载数据集
        
        Args:
            data_dir: 数据目录
            
        Returns:
            DatasetDict: 数据集字典
        """
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "validation.jsonl"
        test_file = data_dir / "test.jsonl"
        
        datasets = {}
        
        for split, file_path in [("train", train_file), ("validation", val_file), ("test", test_file)]:
            if file_path.exists():
                data = self._load_jsonl(file_path)
                dataset = Dataset.from_list(data)
                datasets[split] = dataset
                logger.info(f"加载 {split} 数据集: {len(data)} 条样本")
            else:
                logger.warning(f"文件不存在: {file_path}")
        
        return DatasetDict(datasets)
    
    def _load_jsonl(self, file_path: Path) -> List[Dict]:
        """加载JSONL文件"""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    data.append(item)
        except Exception as e:
            logger.error(f"加载JSONL文件失败 {file_path}: {str(e)}")
        return data
    
    def preprocess_function(self, examples):
        """
        数据预处理函数
        
        Args:
            examples: 样本字典
            
        Returns:
            Dict: 处理后的样本
        """
        # 获取文本和标签
        texts = examples["text"]
        labels = examples.get("label", [0] * len(texts))  # 默认标签为0
        
        # 分词
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        # 添加标签
        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return tokenized
    
    def create_synthetic_data(self, output_dir: Path, num_samples: int = 1000):
        """
        创建合成数据集（用于演示和测试）
        
        Args:
            output_dir: 输出目录
            num_samples: 样本数量
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 示例问题模板
        templates = {
            "车辆信息查询": [
                "{}的型号是什么？",
                "{}的生产日期是哪一年？",
                "{}的配置有哪些？",
                "{}的车辆识别码是多少？",
                "{}的发动机型号是什么？"
            ],
            "操作指南": [
                "如何打开{}的空调？",
                "{}的灯光怎么调节？",
                "如何设置{}的座椅记忆？",
                "{}的雨刷器怎么使用？",
                "如何连接{}的蓝牙？"
            ],
            "故障诊断": [
                "{}启动不了怎么办？",
                "{}的发动机故障灯亮了",
                "{}的刹车有异响",
                "{}的空调不制冷",
                "{}的电池充不进电"
            ],
            "保养维护": [
                "{}多久需要保养一次？",
                "{}的机油怎么更换？",
                "{}的轮胎多久更换？",
                "{}的刹车片磨损标准",
                "{}的空调滤芯更换周期"
            ],
            "安全警告": [
                "{}的安全带提示",
                "{}的胎压报警",
                "{}的ABS故障警告",
                "{}的安全气囊注意事项",
                "{}的儿童锁使用"
            ],
            "技术规格": [
                "{}的百公里加速时间",
                "{}的最大扭矩是多少？",
                "{}的油耗是多少？",
                "{}的电池容量",
                "{}的续航里程"
            ],
            "电池充电": [
                "{}怎么充电？",
                "{}的充电时间要多久？",
                "{}的快充功率",
                "{}的电池保修政策",
                "{}的电池衰减标准"
            ],
            "娱乐系统": [
                "{}的音响怎么调节？",
                "{}的导航系统使用",
                "{}的车载娱乐功能",
                "{}的语音助手怎么用？",
                "{}的屏幕显示设置"
            ],
            "驾驶辅助": [
                "{}的自适应巡航怎么用？",
                "{}的自动泊车功能",
                "{}的车道保持辅助",
                "{}的盲点监测",
                "{}的自动紧急制动"
            ],
            "其他问题": [
                "{}的购车优惠",
                "{}的保险怎么买？",
                "{}的二手车价格",
                "{}的改装建议",
                "{}的驾驶感受"
            ]
        }
        
        # 车型列表
        car_models = [
            "特斯拉Model 3", "宝马X5", "奔驰E级", "奥迪A6", "丰田凯美瑞",
            "本田雅阁", "大众帕萨特", "比亚迪汉", "蔚来ES6", "理想ONE"
        ]
        
        # 生成数据
        train_data = []
        val_data = []
        test_data = []
        
        for intent_idx, (intent_name, intent_templates) in enumerate(templates.items()):
            for template in intent_templates:
                for car_model in car_models:
                    # 生成问题
                    question = template.format(car_model)
                    
                    # 创建样本
                    sample = {
                        "text": question,
                        "label": intent_idx,
                        "intent": intent_name,
                        "car_model": car_model,
                        "template": template
                    }
                    
                    # 按比例分配到不同数据集
                    rand = np.random.random()
                    if rand < 0.7:
                        train_data.append(sample)
                    elif rand < 0.85:
                        val_data.append(sample)
                    else:
                        test_data.append(sample)
        
        # 保存数据
        self._save_jsonl(output_dir / "train.jsonl", train_data[:num_samples//2])
        self._save_jsonl(output_dir / "validation.jsonl", val_data[:num_samples//4])
        self._save_jsonl(output_dir / "test.jsonl", test_data[:num_samples//4])
        
        logger.info(f"生成合成数据集: 训练集 {len(train_data)}，验证集 {len(val_data)}，测试集 {len(test_data)}")
    
    def _save_jsonl(self, file_path: Path, data: List[Dict]):
        """保存JSONL文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

class IntentTrainer:
    """意图分类训练器"""
    
    def __init__(self, config: IntentConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def train(self, dataset: DatasetDict):
        """
        训练意图分类模型
        
        Args:
            dataset: 数据集
        """
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels
        )
        
        # 数据预处理
        tokenized_datasets = dataset.map(
            lambda examples: self._preprocess_function(examples),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # 数据收集器
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # 评估指标
        metric = evaluate.load("accuracy")
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=str(self.config.output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_dir=str(self.config.output_dir / "logs"),
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=3,
            report_to="none"
        )
        
        # 创建训练器
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # 训练模型
        logger.info("开始训练意图分类模型...")
        self.trainer.train()
        
        # 评估模型
        logger.info("评估模型...")
        eval_results = self.trainer.evaluate(tokenized_datasets["test"])
        
        # 保存模型
        self.save_model()
        
        return eval_results
    
    def _preprocess_function(self, examples):
        """数据预处理函数"""
        texts = examples["text"]
        labels = examples["label"]
        
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        tokenized["labels"] = torch.tensor(labels, dtype=torch.long)
        
        return tokenized
    
    def save_model(self):
        """保存模型"""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型和tokenizer（会生成HuggingFace格式的config.json）
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # 将自定义元数据保存到单独文件，不覆盖HuggingFace的config.json
        intent_config_file = self.config.output_dir / "intent_config.json"
        config_data = {
            "model_name": self.config.model_name,
            "num_labels": self.config.num_labels,
            "max_length": self.config.max_length,
            "intent_classes": self.config.INTENT_CLASSES,
            "intent_descriptions": self.config.INTENT_DESCRIPTIONS,
            "training_config": {
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs
            }
        }
        
        with open(intent_config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"模型已保存到: {self.config.output_dir}")
        logger.info(f"意图配置已保存到: {intent_config_file}")

def main():
    """主函数"""
    print("=" * 50)
    print("意图分类模型训练工具 - 汽车座舱RAG系统")
    print("=" * 50)
    
    # 配置
    config = IntentConfig()
    
    # 创建数据集目录
    data_dir = Path(__file__).parent.parent / "data" / "qa_train"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查数据集
    train_file = data_dir / "train.jsonl"
    
    if not train_file.exists():
        print(f"未找到训练数据，创建合成数据集...")
        print(f"数据目录: {data_dir}")
        
        # 创建合成数据
        dataset_creator = IntentDataset(config)
        dataset_creator.create_synthetic_data(data_dir, num_samples=1000)
        
        print("✅ 合成数据集创建完成")
    
    # 加载数据集
    print("📊 加载数据集...")
    dataset_loader = IntentDataset(config)
    dataset = dataset_loader.load_dataset(data_dir)
    
    if "train" not in dataset or len(dataset["train"]) == 0:
        print("❌ 训练数据集为空")
        exit(1)
    
    print(f"训练集: {len(dataset['train'])} 条样本")
    print(f"验证集: {len(dataset['validation'])} 条样本" if "validation" in dataset else "验证集: 0 条样本")
    print(f"测试集: {len(dataset['test'])} 条样本" if "test" in dataset else "测试集: 0 条样本")
    
    # 显示意图类别
    print("\n🎯 意图类别:")
    for idx, intent in enumerate(config.INTENT_CLASSES):
        print(f"  {idx}: {intent} - {config.INTENT_DESCRIPTIONS[intent]}")
    
    # 训练模型
    print("\n🚀 开始训练模型...")
    trainer = IntentTrainer(config)
    
    try:
        eval_results = trainer.train(dataset)
        
        print("\n✅ 训练完成!")
        print(f"模型保存位置: {config.output_dir}")
        
        # 显示评估结果
        if eval_results:
            print("\n📈 评估结果:")
            for metric, value in eval_results.items():
                print(f"  {metric}: {value:.4f}")
        
        # 使用说明
        print("\n📖 使用说明:")
        print("1. 加载训练好的模型:")
        print("   ```python")
        print("   from transformers import AutoTokenizer, AutoModelForSequenceClassification")
        print(f'   tokenizer = AutoTokenizer.from_pretrained("{config.output_dir}")')
        print(f'   model = AutoModelForSequenceClassification.from_pretrained("{config.output_dir}")')
        print("   ```")
        
        print("\n2. 预测意图:")
        print("   ```python")
        print("   from infer_intent import IntentClassifier")
        print(f'   classifier = IntentClassifier("{config.output_dir}")')
        print('   intent, confidence = classifier.predict("如何打开空调？")')
        print('   print(f"意图: {intent}, 置信度: {confidence:.2f}")')
        print("   ```")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()