# 🚗 汽车座舱 RAG 系统

基于检索增强生成（RAG）技术的汽车座舱智能助手系统，支持车辆手册文档处理、语音识别、意图分类和智能问答。

> **系统定位**：面向车辆手册问答场景，用户可输入车辆使用、维护、故障诊断等问题，系统从车辆手册中检索相关内容并生成回答。非汽车相关问题会被自动拦截。

---

## 📁 项目结构

```
car_cockpit_rag/
├── config.py                 # 全局配置（模型、RAG参数）
├── main.py                   # 一键启动入口
├── requirements.txt          # Python依赖清单
│
├── data/                     # 数据目录
│   ├── raw/                  # 原始车辆手册PDF/TXT文件
│   ├── chunks/               # 分块后的JSON文件
│   └── qa_train/             # LoRA微调QA数据集
│
├── data_process/             # 数据处理流水线
│   ├── parse_pdf.py          # PDF解析（pdfplumber）
│   ├── clean_chunk.py        # 文本清洗
│   └── build_chunk.py        # 文本分块（256字符/块）
│
├── vector_store/             # 向量存储
│   ├── build_faiss.py        # FAISS索引构建
│   └── faiss_index/          # 索引文件（.bin + .pkl）
│
├── retriever/                # 检索模块
│   ├── dense_retriever.py    # 稠密向量检索（bge-small-zh-v1.5）
│   ├── bm25_retriever.py     # BM25关键词检索
│   └── rerank.py             # 重排序（CPU环境跳过）
│
├── intent_cls/               # 意图分类
│   ├── train_intent.py       # 训练意图分类器
│   ├── infer_intent.py       # 推理意图分类
│   └── models/               # 微调后的模型
│
├── llm_infer/                # LLM推理
│   ├── rag_llm.py            # RAG管道 + LLM生成 + 领域过滤
│   └── train_lora.py         # LoRA微调脚本
│
├── speech_asr/               # 语音识别
│   └── whisper_asr.py        # Whisper ASR
│
├── gradio_web/               # Web界面
│   └── app.py                # Gradio应用（5个标签页）
│
├── local_model_cache.py      # 本地模型缓存管理
└── models_cache/             # 项目内模型缓存
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
# 推荐使用清华镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 准备模型

项目使用本地缓存模型，无需联网下载：

| 模型 | 用途 | 缓存路径 |
|------|------|----------|
| `bge-small-zh-v1.5` | 文本嵌入 | `models_cache/` 或 `F:/ModelCache/` |
| `Qwen2.5-0.5B-Instruct` | 回答生成 | 同上 |
| 微调BERT | 意图分类（10类） | `intent_cls/models/` |
| `whisper-small` | 语音识别 | `models_cache/` |

> 如需下载模型，设置 HF 镜像：`set HF_ENDPOINT=https://hf-mirror.com`

### 3. 数据处理

将车辆手册 PDF/TXT 文件放入 `data/raw/` 目录，然后执行：

```bash
# 逐步执行
python data_process/parse_pdf.py
python data_process/clean_chunk.py
python data_process/build_chunk.py
python vector_store/build_faiss.py

# 或在Web界面"文档处理"标签页一键运行
```

### 4. 启动系统

```bash
python main.py
```

浏览器访问 **http://localhost:7860**

---

## 🛠️ 功能模块

### 📄 文档处理

- 支持 PDF、TXT、DOCX、MD 格式
- 自动解析 → 清洗 → 分块（256字符/块，30字符重叠）
- 使用 `bge-small-zh-v1.5` 向量化，构建 FAISS 索引

### 🎤 语音识别

- 基于 OpenAI Whisper 的语音转文本
- 支持多种音频格式（wav, mp3, flac 等）

### 🎯 意图分类

- 10 类车辆相关意图识别（故障诊断、保养维护、电池充电等）
- 基于 `bge-small-zh-v1.5` 微调的 BERT 分类器
- 支持自定义训练（`train_intent.py`）

### 📚 智能检索

- **稠密检索**：`bge-small-zh-v1.5` + FAISS，擅长语义相似度
- **BM25检索**：`rank_bm25`，擅长关键词精确匹配
- 混合检索 + 分数合并去重，CPU 环境跳过重排序（节省30-60秒）

### 💬 智能问答

- 基于 RAG 的问答系统
- **领域过滤**：自动拒绝非汽车相关问题（关键词 + Prompt 双层防护）
- 精简 Prompt + 截断文档，适配 CPU 环境
- 支持 LoRA 微调提升领域问答能力

---

## 🔧 配置说明

### 模型配置（`config.py` → `MODEL_CONFIG`）

```python
"embedding_model": "BAAI/bge-small-zh-v1.5"   # 嵌入模型（512维）
"embedding_dim": 512
"llm_model": "Qwen/Qwen2.5-0.5B-Instruct"     # LLM模型（CPU环境）
"asr_model": "openai/whisper-small"             # 语音识别
"intent_model": "BAAI/bge-small-zh-v1.5"       # 意图分类基座
```

### RAG 配置（`config.py` → `RAG_CONFIG`）

```python
"chunk_size": 256        # 分块大小（字符数），适配简短问题检索
"chunk_overlap": 30      # 分块重叠
"top_k": 3               # 检索返回数量
"rerank_top_k": 2        # 重排序后保留数量
"temperature": 0.7       # 生成温度
"max_new_tokens": 150    # 最大生成token数（CPU优化）
```

### 模型缓存配置

```python
"model_download": {
    "use_mirror": True,
    "mirror_url": "https://hf-mirror.com",
    "use_local_cache": True,
    "local_cache_path": "F:/ModelCache/huggingface/hub",  # 主缓存
    "cache_dir": "models_cache",                           # 项目缓存
}
```

---

## 🌐 Web 界面

系统提供 5 个功能标签页：

| 标签页 | 功能 |
|--------|------|
| 📄 文档处理 | 上传车辆手册，运行数据处理流水线 |
| 🎤 语音转录 | 上传音频文件，转录为文本 |
| 🎯 意图分析 | 输入查询，显示意图分类结果和置信度 |
| 📚 文档检索 | 输入查询，检索相关文档片段 |
| 💬 智能问答 | 对话界面，支持示例问题 |

---

## 🎯 意图类别

| 编号 | 意图 | 示例问题 |
|------|------|----------|
| 0 | 车辆信息查询 | "车型号是什么？" |
| 1 | 操作指南 | "怎么打开空调？" |
| 2 | 故障诊断 | "车辆启动不了" |
| 3 | 保养维护 | "多久保养一次？" |
| 4 | 安全警告 | "安全带提示" |
| 5 | 技术规格 | "百公里加速时间" |
| 6 | 电池充电 | "怎么充电？" |
| 7 | 娱乐系统 | "音响怎么调？" |
| 8 | 驾驶辅助 | "自适应巡航怎么用？" |
| 9 | 其他问题 | 不属于以上类别 |

---

## 📊 数据流水线

```
原始PDF → PDF解析 → 文本清洗 → 文本分块 → 向量化 → FAISS索引
  ↓           ↓           ↓           ↓          ↓          ↓
data/raw/  _parsed.json _cleaned.json _chunked.json  512维向量  faiss_index.bin
```

---

## 🐛 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 模型加载失败 | 缓存路径不正确 | 检查 `config.py` 中 `local_cache_path` |
| 检索无结果 | FAISS 索引未构建 | 运行数据处理流水线 |
| 回答很慢 | CPU 推理 | 已优化：减少token数、精简Prompt、跳过重排序 |
| 非汽车问题也被回答 | 缺少领域过滤 | 已添加关键词过滤 + Prompt 约束 |
| `SentenceTransformer.from_pretrained` 报错 | 5.x 版本 API 变更 | 使用 `SentenceTransformer(model_name)` 构造 |
| 意图分类器加载失败 | config.json 被覆盖 | 从 checkpoint 恢复或重新训练 |
| 依赖安装失败 | 网络问题 | 使用清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple` |

---

## 📈 性能优化

### 硬件要求

- **最低**: 8GB RAM, 4核 CPU
- **推荐**: 16GB RAM, 8核 CPU, GPU（用于 LLM 推理）

### 已实施的 CPU 优化

| 优化项 | 修改前 | 修改后 | 效果 |
|--------|--------|--------|------|
| 生成 token 数 | 512 | 150 | 生成时间减少 70% |
| Prompt 长度 | 长系统提示 + 全文档 | 精简 + 截断 200字/篇 | 输入 token 减少 60% |
| 检索量 | top_k×2=10 | top_k=3 | 检索耗时减半 |
| 重排序 | 加载 bge-reranker | CPU 环境跳过 | 省 30-60 秒 |
| 分块大小 | 512 字符 | 256 字符 | 检索更精准 |
| 领域过滤 | 无 | 关键词 + Prompt 双层 | 拒绝无关问题 |

---

## 📚 相关文档

- [学习手册.md](学习手册.md) — 完整系统学习手册（含架构图）
- [启动说明.md](启动说明.md) — 启动指南
- [本地缓存使用指南.md](本地缓存使用指南.md) — 模型缓存配置

---

## 🙏 致谢

- [Hugging Face](https://huggingface.co/) — 预训练模型
- [FAISS](https://github.com/facebookresearch/faiss) — 向量相似度搜索
- [Gradio](https://www.gradio.app/) — Web 界面框架
- [Whisper](https://github.com/openai/whisper) — 语音识别
- [Qwen](https://github.com/QwenLM/Qwen) — 大语言模型
- [BGE](https://github.com/FlagOpen/FlagEmbedding) — 文本嵌入

---

**注意**: 本项目为演示用途，实际部署时请根据需求调整配置和安全设置。
