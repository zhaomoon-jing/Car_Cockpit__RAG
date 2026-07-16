# 数据目录说明

此目录存放汽车座舱RAG系统的数据文件。

## 目录结构

### raw/
存放原始的车辆手册文件，支持以下格式：
- PDF文件 (*.pdf)
- 文本文件 (*.txt)
- Word文档 (*.docx)
- Markdown文件 (*.md)

**使用建议：**
1. 将车辆用户手册、维修手册等PDF文件放入此目录
2. 支持中文和英文文档
3. 建议文件命名规范：`车型_手册类型_版本.pdf`
   - 例如：`Tesla_Model3_用户手册_v2.1.pdf`
   - 例如：`BMW_X5_维修手册_v1.0.pdf`

### chunks/
存放文档分块后的JSON文件，每个JSON文件包含：
- 文本块内容
- 元数据（来源文件、页码、位置等）
- 嵌入向量（可选）

**文件格式：**
```json
{
  "chunks": [
    {
      "id": "unique_id",
      "text": "文本内容...",
      "metadata": {
        "source": "文件名.pdf",
        "page": 1,
        "section": "概述",
        "chunk_index": 0
      },
      "embedding": [0.1, 0.2, ...]  // 512维向量
    }
  ]
}
```

### qa_train/
存放LoRA微调用的QA数据集，用于训练车辆相关的问答模型。

**文件格式：**
- `train.jsonl`: 训练数据
- `validation.jsonl`: 验证数据
- `test.jsonl`: 测试数据

**JSONL格式示例：**
```json
{
  "instruction": "特斯拉Model 3的电池保修政策是什么？",
  "input": "根据特斯拉车辆手册",
  "output": "特斯拉为Model 3提供8年或16万公里的电池质保..."
}
```

## 数据预处理流程

1. **文档上传**：将车辆手册放入 `raw/` 目录
2. **文档解析**：运行 `python data_process/parse_pdf.py`
3. **文本分块**：运行 `python data_process/build_chunk.py`
4. **向量化**：运行 `python vector_store/build_faiss.py`

## 注意事项

1. **文件编码**：确保文本文件使用UTF-8编码
2. **文件大小**：单个PDF文件建议不超过50MB
3. **数据安全**：不要存放敏感信息
4. **版本控制**：建议使用git管理文档版本

## 数据示例

在 `raw/` 目录中可以放置以下类型的文档：
- 车辆用户手册
- 维修保养手册
- 技术规格说明书
- 常见问题解答(FAQ)
- 故障排除指南

## 相关脚本

- `data_process/parse_pdf.py`: PDF解析脚本
- `data_process/clean_chunk.py`: 文本清洗脚本
- `data_process/build_chunk.py`: 分块构建脚本