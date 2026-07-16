# 🚗 汽车座舱RAG系统

基于检索增强生成技术的汽车座舱智能助手系统，支持车辆手册文档处理、语音识别、意图分类和智能问答。

## 📁 项目结构

```
car_cockpit_rag/
├── config.py                 # 全局配置
├── requirements.txt          # Python依赖包
├── main.py                   # 一键启动Gradio
├── data/                     # 数据目录
│   ├── raw/                  # 原始车辆手册PDF/TXT文件
│   ├── chunks/               # 分块后的JSON文件
│   └── qa_train/             # LoRA微调QA数据集
├── data_process/             # 数据处理模块
│   ├── parse_pdf.py          # PDF解析脚本
│   ├── clean_chunk.py        # 文本清洗脚本
│   └── build_chunk.py        # 文本分块脚本
├── vector_store/             # 向量存储模块
│   ├── faiss_index/          # FAISS向量索引文件
│   └── build_faiss.py        # FAISS索引构建脚本
├── speech_asr/               # 语音识别模块
│   └── whisper_asr.py        # Whisper语音识别脚本
├── intent_cls/               # 意图分类模块
│   ├── train_intent.py       # 意图分类训练脚本
│   └── infer_intent.py       # 意图分类推理脚本
├── retriever/                # 检索器模块
│   ├── bm25_retriever.py     # BM25检索器
│   ├── dense_retriever.py    # 稠密检索器
│   └── rerank.py             # 重排序器
├── llm_infer/                # LLM推理模块
│   ├── train_lora.py         # LoRA微调脚本
│   └── rag_llm.py            # LLM RAG推理脚本
└── gradio_web/               # Web界面
    └── app.py                # Gradio可视化页面
```

## 🚀 快速开始

### ⚠️ 重要提示：HuggingFace连接问题

如果遇到 `Connection to huggingface.co timed out` 错误，请先运行：

```bash
# 方法1：使用本地缓存（推荐已有模型的用户）
python use_local_models.py

# 方法2：设置本地缓存环境
python setup_local_cache.py

# 方法3：使用简化版本（推荐新手）
python main_simple.py

# 方法4：修复连接问题
python fix_huggingface.py

# 方法5：下载小模型
python download_models.py
```

详细解决方案见：
- [解决HuggingFace连接问题.md](解决HuggingFace连接问题.md)
- [本地缓存使用指南.md](本地缓存使用指南.md)

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将车辆手册PDF/TXT文件放入 `data/raw/` 目录。

### 3. 数据处理流程

```bash
# 1. 解析PDF文档
python data_process/parse_pdf.py

# 2. 清洗文本
python data_process/clean_chunk.py

# 3. 文本分块
python data_process/build_chunk.py

# 4. 构建向量索引
python vector_store/build_faiss.py

# 5. 训练意图分类模型（可选）
python intent_cls/train_intent.py

# 6. 训练LoRA模型（可选）
python llm_infer/train_lora.py
```

### 4. 启动Web界面

```bash
python main.py
```

或直接运行：

```bash
python gradio_web/app.py
```

访问 http://localhost:7860 使用Web界面。

## 🛠️ 功能模块

### 📄 文档处理
- 支持PDF、TXT、DOCX、MD格式
- 自动解析、清洗、分块
- 构建FAISS向量索引

### 🎤 语音识别
- 基于Whisper的语音转文本
- 支持多种音频格式
- 实时转录功能

### 🎯 意图分类
- 10类车辆相关意图识别
- 基于BERT的意图分类模型
- 支持自定义训练

### 📚 智能检索
- BM25关键词检索
- 稠密向量检索（FAISS）
- 混合检索与重排序

### 💬 智能问答
- 基于RAG的问答系统
- 支持LoRA微调
- 上下文感知回答

## 🔧 配置说明

### 模型配置 (`config.py`)

```python
MODEL_CONFIG = {
    "embedding_model": "BAAI/bge-small-zh-v1.5",  # 嵌入模型
    "llm_model": "Qwen/Qwen2.5-7B-Instruct",      # LLM模型
    "asr_model": "openai/whisper-small",          # 语音识别模型
    "intent_model": "bert-base-chinese",          # 意图分类模型
}
```

### RAG配置 (`config.py`)

```python
RAG_CONFIG = {
    "chunk_size": 512,        # 文本分块大小
    "chunk_overlap": 50,      # 分块重叠大小
    "top_k": 5,               # 检索结果数
    "rerank_top_k": 3,        # 重排序结果数
    "temperature": 0.7,       # 生成温度
    "max_new_tokens": 512,    # 最大生成长度
}
```

## 📊 数据处理流程

1. **文档上传** → `data/raw/`
2. **PDF解析** → 提取文本和元数据
3. **文本清洗** → 移除噪音和格式化
4. **文本分块** → 语义分块处理
5. **向量化** → 构建FAISS索引
6. **模型训练** → 意图分类和LoRA微调
7. **检索问答** → 基于RAG的智能问答

## 🎯 意图类别

系统支持10类车辆相关意图：
1. **车辆信息查询** - 查询车辆基本信息
2. **操作指南** - 如何操作车辆功能
3. **故障诊断** - 车辆故障相关问题
4. **保养维护** - 保养和维护问题
5. **安全警告** - 安全相关警告和提示
6. **技术规格** - 技术参数和规格
7. **电池充电** - 电池和充电问题
8. **娱乐系统** - 娱乐和信息系统
9. **驾驶辅助** - 驾驶辅助功能
10. **其他问题** - 其他类型问题

## 🌐 Web界面功能

### 标签页说明

1. **📄 文档处理** - 上传和处理车辆手册
2. **🎤 语音转录** - 语音转文本功能
3. **🎯 意图分析** - 分析用户查询意图
4. **📚 文档检索** - 检索相关文档
5. **💬 智能问答** - 与车辆手册对话
6. **⚙️ 系统状态** - 查看系统配置和状态

## 🔍 高级功能

### 混合检索
结合BM25关键词检索和稠密向量检索，提供更准确的搜索结果。

### 重排序
使用重排序模型对检索结果进行优化，提升相关性。

### LoRA微调
使用LoRA技术对LLM进行微调，提升领域特定问答能力。

### 意图感知
根据用户意图调整检索策略和回答风格。

## 🐛 故障排除

### 常见问题

1. **依赖安装失败**
   ```bash
   # 使用国内镜像
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. **模型下载慢**
   ```bash
   # 设置HF镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

3. **内存不足**
   - 减小 `chunk_size`
   - 使用更小的嵌入模型
   - 启用GPU加速

### 日志查看

```bash
# 查看系统日志
tail -f logs/car_cockpit_rag.log
```

## 📈 性能优化

### 硬件要求
- **最低**: 8GB RAM, 4核CPU
- **推荐**: 16GB RAM, 8核CPU, GPU (用于LLM推理)

### 优化建议
1. 使用更小的嵌入模型
2. 减小文本分块大小
3. 使用量化模型
4. 启用GPU加速

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [Hugging Face](https://huggingface.co/) - 提供预训练模型
- [FAISS](https://github.com/facebookresearch/faiss) - 向量相似度搜索
- [Gradio](https://www.gradio.app/) - 快速构建Web界面
- [Whisper](https://github.com/openai/whisper) - 语音识别模型

## 📞 支持

如有问题或建议，请提交 [Issue](https://github.com/your-repo/issues)。

---

**注意**: 本项目为演示用途，实际部署时请根据需求调整配置和安全设置。