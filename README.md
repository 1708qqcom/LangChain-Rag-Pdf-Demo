# LangChain-Rag-Pdf-Demo
使用LangChain，将pdf内容读取，用langchain-text-splitters切割内容，结合chromadb向量数据，保存数据，智能检索数据

# Quick Start
```
1.安装相应的python库，
pip install -r requirements.txt
2.运行代码
python LangChain_Agent.py
```

# 实现效果
 <img width="1055" height="508" alt="image" src="https://github.com/user-attachments/assets/98678d15-9e59-445e-8020-3abe017ef928" />
能很好的匹配到对应的pdf文段
<img width="1096" height="1002" alt="image" src="https://github.com/user-attachments/assets/d9f45035-7b0e-4a83-a1f0-10404a975aad" />

# 检测是否支持GPU运行
```
python gpu_detection.py
```
<img width="1056" height="416" alt="image" src="https://github.com/user-attachments/assets/2239cdae-d492-4e68-92d4-3a02cd4d62b9" />

