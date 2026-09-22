# 医知助手 · 医疗健康 RAG Agent

一个基于 **LangGraph + LangChain** 的医疗健康检索增强生成（RAG）Agent 项目，支持多工具调用、混合检索、重排精排、图片 OCR 读图与网页交互界面。用于展示 RAG + Agent 的完整工程落地能力。

## 核心亮点

- **多工具 ReAct Agent**：知识检索、紧急分级、剂量计算、单位换算，由 LLM 自主决策是否调用。
- **混合检索（多路召回）**：语义向量（BGE + Chroma）与 BM25 双路召回，RRF 融合。
- **Rerank 重排**：cross-encoder 二次精排 + 相关性阈值过滤，剔除无关来源。
- **真 ReAct 决策**：寒暄不调库、医疗问题才检索，避免"伪 ReAct"固定流程。
- **图片 OCR**：上传化验单/报告图片，自动提取文字入库并参与检索。
- **用户自带 Key（BYOK）**：前端填写自己的 DeepSeek Key，Key 存 sessionStorage，本地回环转发，不写入服务端。
- **多轮记忆**：累积历史 + 摘要压缩 + 截断保底。

## 快速开始

```bash
# 1. 安装依赖（建议先建虚拟环境）
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-rag.txt

# 2. 设置国内模型镜像（可跳过，若下载慢）
set HF_ENDPOINT=https://hf-mirror.com

# 3. 配置 Key（二选一：填 .env 或网页端自行填写）
copy .env.example .env

# 4. 构建知识库
python 01_建知识库.py

# 5. 启动网页版
cd web
python server.py
# 浏览器打开 http://127.0.0.1:8000
```

## 命令行示例

```bash
python 04_react_agent.py   # 多工具 ReAct Agent（终端交互）
python 07_混合检索评测.py   # 混合检索 vs 纯向量 对比
python 09_rerank评测.py    # Rerank 重排效果对比
python 06_复盘react.py     # 验证 Agent 是否为真 ReAct
python 05_ocr入库.py       # 图片 OCR 文字入库
```

## 技术栈

Python · LangGraph · LangChain · Chroma · sentence-transformers · BGE（bge-small-zh-v1.5 / bge-reranker-base）· BM25 · RapidOCR · DeepSeek · 原生 HTTP Server（零 Web 依赖）