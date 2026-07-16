#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA微调模块 - 使用LoRA技术微调大语言模型
"""

import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass
from tqdm import tqdm

# 导入项目配置
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODEL_CONFIG, QA_TRAIN_DIR

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class LoRAConfig:
    """LoRA配置"""
    model_name: str = MODEL_CONFIG['llm_model']
    lora_rank: int = MODEL_CONFIG['lora_rank']
    lora_alpha: int = MODEL_CONFIG['lora_alpha']
    lora_dropout: float = MODEL_CONFIG['lora_dropout']
    
    # 训练参数
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_length: int = 512
    
    # 输出路径
    output_dir: Path = Path(__file__).parent / "lora_models"
    logging_dir: Path = Path(__file__).parent / "logs"
    
    # 数据集路径
    train_file: Path = QA_TRAIN_DIR / "train.jsonl"
    val_file: Path = QA_TRAIN_DIR / "validation.jsonl"
    test_file: Path = QA_TRAIN_DIR / "test.jsonl"

class QADataset:
    """QA数据集"""
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.tokenizer = None
        
    def load_dataset(self) -> Dict[str, List[Dict]]:
        """
        加载QA数据集
        
        Returns:
            Dict[str, List[Dict]]: 数据集字典
        """
        datasets = {}
        
        for split, file_path in [("train", self.config.train_file), 
                                ("validation", self.config.val_file), 
                                ("test", self.config.test_file)]:
            if file_path.exists():
                data = self._load_jsonl(file_path)
                datasets[split] = data
                logger.info(f"加载 {split} 数据集: {len(data)} 条样本")
            else:
                logger.warning(f"文件不存在: {file_path}")
                datasets[split] = []
        
        return datasets
    
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
    
    def create_synthetic_data(self, num_samples: int = 1000):
        """
        创建合成QA数据集（用于演示和测试）
        
        Args:
            num_samples: 样本数量
        """
        # 确保目录存在
        self.config.train_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 车辆相关QA模板
        qa_templates = [
            {
                "instruction": "根据车辆手册，回答以下问题：",
                "input_template": "问题：{}",
                "output_template": "根据车辆手册，{}"
            },
            {
                "instruction": "请根据车辆技术文档提供准确信息：",
                "input_template": "用户询问：{}",
                "output_template": "根据技术文档，{}"
            },
            {
                "instruction": "基于车辆使用说明，回答用户问题：",
                "input_template": "问题描述：{}",
                "output_template": "使用说明指出，{}"
            }
        ]
        
        # 车辆相关问题和答案
        qa_pairs = [
            # 车辆信息
            ("特斯拉Model 3的电池容量是多少？", "特斯拉Model 3的电池容量根据配置不同，标准续航版为60千瓦时，长续航版为75千瓦时，高性能版为82千瓦时。"),
            ("宝马X5的发动机功率是多少？", "宝马X5搭载的3.0T直列六缸发动机最大功率为250千瓦（340马力），最大扭矩为450牛·米。"),
            ("奔驰E级的轴距是多少？", "奔驰E级长轴距版的轴距为3079毫米，为标准版轴距加长了140毫米。"),
            
            # 操作指南
            ("怎么打开特斯拉的空调？", "在特斯拉中控屏幕上点击温度图标，然后调节温度滑块，或使用语音命令说'打开空调'。"),
            ("如何设置宝马的座椅记忆？", "调整好座椅位置后，按住车门上的SET按钮，再按1、2或3中的任意一个数字键，听到提示音即设置成功。"),
            ("奔驰的自动泊车怎么用？", "车速低于30公里/小时时，系统会自动扫描停车位，找到车位后根据屏幕提示挂入R挡，松开方向盘和刹车即可。"),
            
            # 故障诊断
            ("车辆无法启动怎么办？", "首先检查电池电量，如果电量不足需要充电或搭电；其次检查钥匙是否在车内；最后检查挡位是否在P挡。"),
            ("发动机故障灯亮了怎么办？", "立即减速并安全停车，联系专业维修人员检查。可能是传感器故障、排放系统问题或发动机本身故障。"),
            ("刹车有异响怎么处理？", "可能是刹车片磨损到极限，需要更换；或者是刹车盘上有异物，需要清理；建议尽快到维修店检查。"),
            
            # 保养维护
            ("电动汽车多久保养一次？", "电动汽车建议每1年或2万公里进行一次常规保养，主要检查电池、电机、刹车系统和轮胎。"),
            ("机油多久更换一次？", "传统燃油车一般每6个月或5000-10000公里更换一次机油，具体参考车辆使用手册。"),
            ("轮胎气压多少合适？", "一般轿车胎压在2.3-2.5bar之间，具体数值在驾驶座车门B柱或油箱盖内侧有标注。"),
            
            # 安全警告
            ("安全气囊注意事项有哪些？", "不要在前排放置儿童安全座椅；不要遮挡气囊展开区域；发生碰撞后即使气囊未展开也应检查系统。"),
            ("ABS故障警告灯亮了怎么办？", "立即减速慢行，避免急刹车，尽快到维修店检查ABS系统，可能是传感器故障或系统问题。"),
            ("胎压报警怎么处理？", "立即停车检查轮胎，如有漏气更换备胎或补胎；如无漏气可能是胎压传感器故障，需要到维修店检查。"),
            
            # 电池充电
            ("电动汽车怎么充电？", "使用随车充电器连接家用电源，或到公共充电站使用快充桩。充电前确保充电口干燥清洁。"),
            ("快充对电池有损害吗？", "偶尔使用快充影响不大，但长期频繁使用快充可能加速电池衰减。建议以慢充为主，快充为辅。"),
            ("电池保修多久？", "多数电动汽车电池质保为8年或16万公里，具体以厂家政策为准，特斯拉为8年或16-24万公里。"),
            
            # 娱乐系统
            ("怎么连接车载蓝牙？", "在车机设置中打开蓝牙，手机搜索车辆蓝牙名称并配对，输入配对码（通常为0000或1234）。"),
            ("导航系统怎么更新？", "通过车载系统连接Wi-Fi自动更新，或到官网下载更新包用U盘安装，建议每半年更新一次。"),
            ("语音助手怎么唤醒？", "大多数车辆通过方向盘上的语音按钮或说出唤醒词（如'你好，奔驰'、'嗨，宝马'）来唤醒语音助手。"),
            
            # 驾驶辅助
            ("自适应巡航怎么用？", "按下方向盘上的ACC按钮，设置 desired speed，车辆会自动保持与前车的安全距离。"),
            ("车道保持辅助会主动转向吗？", "当车辆偏离车道时，系统会通过轻微转向干预或振动提醒，但驾驶员仍需手握方向盘。"),
            ("自动紧急制动什么时候触发？", "当系统检测到与前方车辆或行人可能发生碰撞且驾驶员未采取制动措施时，会自动全力制动。"),
        ]
        
        # 生成数据
        train_data = []
        val_data = []
        test_data = []
        
        for i in range(num_samples):
            # 随机选择模板和QA对
            template = np.random.choice(qa_templates)
            question, answer = np.random.choice(qa_pairs)
            
            # 随机选择车型
            car_models = ["特斯拉Model 3", "宝马X5", "奔驰E级", "奥迪A6", "丰田凯美瑞"]
            car_model = np.random.choice(car_models)
            
            # 替换车型
            question = question.replace("特斯拉", car_model.split()[0]).replace("Model 3", car_model)
            answer = answer.replace("特斯拉", car_model.split()[0]).replace("Model 3", car_model)
            
            # 构建样本
            sample = {
                "instruction": template["instruction"],
                "input": template["input_template"].format(question),
                "output": template["output_template"].format(answer),
                "car_model": car_model,
                "qa_type": "synthetic",
                "id": f"syn_{i}"
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
        self._save_jsonl(self.config.train_file, train_data)
        self._save_jsonl(self.config.val_file, val_data)
        self._save_jsonl(self.config.test_file, test_data)
        
        logger.info(f"生成合成数据集: 训练集 {len(train_data)}，验证集 {len(val_data)}，测试集 {len(test_data)}")
    
    def _save_jsonl(self, file_path: Path, data: List[Dict]):
        """保存JSONL文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

class LoRATrainer:
    """LoRA训练器"""
    
    def __init__(self, config: LoRAConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def prepare_model(self):
        """准备模型和tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            
            logger.info(f"正在加载模型: {self.config.model_name}")
            
            # 配置4-bit量化
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 准备模型用于k-bit训练
            self.model = prepare_model_for_kbit_training(self.model)
            
            # 配置LoRA
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            
            # 应用LoRA
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            
            logger.info("模型准备完成")
            
        except ImportError as e:
            logger.error(f"导入失败: {str(e)}")
            logger.error("请安装: pip install transformers accelerate peft bitsandbytes")
            raise
        except Exception as e:
            logger.error(f"准备模型失败: {str(e)}")
            raise
    
    def prepare_dataset(self, datasets: Dict[str, List[Dict]]):
        """准备数据集"""
        try:
            from datasets import Dataset, DatasetDict
            
            # 转换格式
            def format_example(example):
                # 构建prompt
                prompt = f"{example['instruction']}\n\n{example['input']}\n\n"
                completion = example['output']
                
                return {
                    "text": prompt + completion,
                    "prompt": prompt,
                    "completion": completion
                }
            
            # 创建Dataset
            formatted_datasets = {}
            for split, data in datasets.items():
                if data:
                    formatted_data = [format_example(item) for item in data]
                    formatted_datasets[split] = Dataset.from_list(formatted_data)
            
            dataset_dict = DatasetDict(formatted_datasets)
            
            # Tokenize
            def tokenize_function(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
            
            tokenized_datasets = dataset_dict.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset_dict["train"].column_names if "train" in dataset_dict else []
            )
            
            return tokenized_datasets
            
        except Exception as e:
            logger.error(f"准备数据集失败: {str(e)}")
            raise
    
    def train(self, datasets: Dict[str, List[Dict]]):
        """
        训练LoRA模型
        
        Args:
            datasets: 数据集
        """
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            # 准备模型
            self.prepare_model()
            
            # 准备数据集
            tokenized_datasets = self.prepare_dataset(datasets)
            
            if "train" not in tokenized_datasets or len(tokenized_datasets["train"]) == 0:
                logger.error("训练数据集为空")
                return
            
            # 数据收集器
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # 训练参数
            training_args = TrainingArguments(
                output_dir=str(self.config.output_dir),
                overwrite_output_dir=True,
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                logging_dir=str(self.config.logging_dir),
                logging_steps=10,
                save_strategy="epoch",
                evaluation_strategy="epoch" if "validation" in tokenized_datasets else "no",
                save_total_limit=3,
                load_best_model_at_end=True if "validation" in tokenized_datasets else False,
                metric_for_best_model="loss",
                greater_is_better=False,
                fp16=True,
                report_to="none"
            )
            
            # 创建训练器
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets.get("validation", None),
                data_collator=data_collator,
                tokenizer=self.tokenizer,
            )
            
            # 训练模型
            logger.info("开始训练LoRA模型...")
            self.trainer.train()
            
            # 保存模型
            self.save_model()
            
            # 评估模型
            if "test" in tokenized_datasets and len(tokenized_datasets["test"]) > 0:
                logger.info("评估模型...")
                eval_results = self.trainer.evaluate(tokenized_datasets["test"])
                logger.info(f"测试集损失: {eval_results.get('eval_loss', 'N/A')}")
            
            return self.trainer.state.log_history
            
        except Exception as e:
            logger.error(f"训练失败: {str(e)}")
            raise
    
    def save_model(self):
        """保存模型"""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # 保存配置
        config_file = self.config.output_dir / "config.json"
        config_data = {
            "base_model": self.config.model_name,
            "lora_rank": self.config.lora_rank,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "training_config": {
                "batch_size": self.config.batch_size,
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "max_length": self.config.max_length
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"LoRA模型已保存到: {self.config.output_dir}")

def main():
    """主函数"""
    print("=" * 50)
    print("LoRA微调工具 - 汽车座舱RAG系统")
    print("=" * 50)
    
    # 配置
    config = LoRAConfig()
    
    # 检查数据集
    if not config.train_file.exists():
        print(f"未找到训练数据，创建合成数据集...")
        print(f"数据目录: {config.train_file.parent}")
        
        # 创建合成数据
        dataset_creator = QADataset(config)
        dataset_creator.create_synthetic_data(num_samples=1000)
        
        print("✅ 合成数据集创建完成")
    
    # 加载数据集
    print("📊 加载数据集...")
    dataset_loader = QADataset(config)
    datasets = dataset_loader.load_dataset()
    
    if "train" not in datasets or len(datasets["train"]) == 0:
        print("❌ 训练数据集为空")
        exit(1)
    
    print(f"训练集: {len(datasets['train'])} 条样本")
    print(f"验证集: {len(datasets['validation'])} 条样本" if "validation" in datasets else "验证集: 0 条样本")
    print(f"测试集: {len(datasets['test'])} 条样本" if "test" in datasets else "测试集: 0 条样本")
    
    # 显示示例
    print("\n📝 数据示例:")
    for i, sample in enumerate(datasets["train"][:3]):
        print(f"\n示例 {i+1}:")
        print(f"  指令: {sample.get('instruction', 'N/A')}")
        print(f"  输入: {sample.get('input', 'N/A')}")
        print(f"  输出: {sample.get('output', 'N/A')[:100]}...")
    
    # 训练模型
    print("\n🚀 开始训练LoRA模型...")
    print(f"模型: {config.model_name}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"LoRA alpha: {config.lora_alpha}")
    print(f"训练轮数: {config.num_epochs}")
    print(f"批大小: {config.batch_size}")
    
    trainer = LoRATrainer(config)
    
    try:
        history = trainer.train(datasets)
        
        print("\n✅ 训练完成!")
        print(f"模型保存位置: {config.output_dir}")
        
        # 显示训练统计
        if history:
            print("\n📈 训练统计:")
            train_losses = [log['loss'] for log in history if 'loss' in log]
            if train_losses:
                print(f"  最终训练损失: {train_losses[-1]:.4f}")
            
            eval_losses = [log['eval_loss'] for log in history if 'eval_loss' in log]
            if eval_losses:
                print(f"  最终验证损失: {eval_losses[-1]:.4f}")
        
        # 使用说明
        print("\n📖 使用说明:")
        print("1. 加载训练好的LoRA模型:")
        print("   ```python")
        print("   from peft import PeftModel")
        print("   from transformers import AutoTokenizer, AutoModelForCausalLM")
        print("   ")
        print(f'   tokenizer = AutoTokenizer.from_pretrained("{config.model_name}")')
        print(f'   model = AutoModelForCausalLM.from_pretrained("{config.model_name}")')
        print(f'   model = PeftModel.from_pretrained(model, "{config.output_dir}")')
        print("   ```")
        
        print("\n2. 在RAG中使用:")
        print("   ```python")
        print("   from rag_llm import RAGLLM")
        print(f'   rag_llm = RAGLLM(lora_model_path="{config.output_dir}")')
        print('   answer = rag_llm.generate("特斯拉的电池容量是多少？", context="根据手册...")')
        print('   print(answer)')
        print("   ```")
        
    except Exception as e:
        print(f"❌ 训练失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()