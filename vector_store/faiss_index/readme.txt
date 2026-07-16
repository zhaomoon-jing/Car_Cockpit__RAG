📖 使用说明:
1. 索引文件:
   • faiss_index.bin - FAISS索引文件
   • metadata.pkl - 文本块元数据
   • config.json - 索引配置

2. 在代码中使用索引:
   ```python
   import faiss
   import pickle

   # 加载索引
   index = faiss.read_index("F:\car_cockpit_rag\vector_store\faiss_index/faiss_index.bin")
   with open("F:\car_cockpit_rag\vector_store\faiss_index/metadata.pkl", "rb") as f:
       metadata = pickle.load(f)
   ```