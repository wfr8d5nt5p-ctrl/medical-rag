# 医知助手 · 医疗健康 RAG Agent

一个基于 **LangGraph + LangChain** 的医疗健康检索增强生成（RAG）Agent 项目，支持多工具调用、混合检索、重排精排、图片 OCR 读图与网页交互界面。用于展示 RAG + Agent 的完整工程落地能力。

## 项目亮点

- **多工具 ReAct Agent**：集成知识检索、紧急分级、剂量计算、单位换算四个可调用工具，由 LLM 通过"思考→行动→观察"循环自主决策工具调用，而非预设固定流程。
- **真 ReAct 决策**：采用 LangGraph 状态图编排，寒暄/非医疗问题不触发检索，医疗问题才调用知识库，从行为层面验证了 Agent 的真实决策能力（配套 `06_复盘react.py` 回归测试）。
- **混合检索（多路召回 + RRF 融合）**：语义向量（BGE 中文嵌入 + Chroma）与 BM25 关键词双路并行召回，经 RRF（倒数排名融合）合并，兼顾语义相似与字面命中，提升同义词/术语/口语场景的召回率（配套 `07_混合检索评测.py` 对比）。
- **Rerank 重排 + 阈值过滤**：粗召回 Top-N 后由 cross-encoder（bge-reranker）逐对精排，并按置信度阈值剔除低相关来源，解决"答案误解读/答非所问"，让返回给用户的知识更可信（配套 `09_rerank评测.py` 对比）。
- **图片 OCR 读取**：上传化验单/报告截图，Pillow 字节流解码绕开后缀猜测问题，RapidOCR 提取文字写入知识库，使图片内容可被跨会话检索。
- **多轮记忆 + 压缩**：对话历史累积，超过阈值时用 LLM 压缩早期上下文为摘要，控制上下文窗口、保留关键病情信息。
- **用户自带 Key（BYOK）**：前端填写个人 DeepSeek Key，存 sessionStorage（关标签页即清除），本地回环转发至后端，服务端不硬编码密钥、强校验缺失即拒绝，兼顾隐私与部署成本。

## 核心亮点（速览）

- 多工具 ReAct Agent · 真决策 · 混合检索（向量+BM25+RRF）· Rerank 精排 · OCR 读图 · 多轮记忆 · BYOK

## 快速开始

```bash
# 1. 创建 uv 虚拟环境并安装依赖
#    （未安装 uv 时可先用 python -m pip install uv 安装）
uv venv .venv
uv pip install -r requirements-rag.txt

# 2. 设置国内模型镜像（可跳过，若下载慢）
set HF_ENDPOINT=https://hf-mirror.com

# 3. 配置 Key（二选一：填 .env 或网页端自行填写）
copy .env.example .env

# 4. 构建知识库
.venv\Scripts\python 01_建知识库.py

# 5. 启动网页版
cd web
.venv\Scripts\python server.py
# 浏览器打开 http://127.0.0.1:8000
```

> 也可以直接双击 `启动医知助手.bat`，脚本会优先使用 `.venv` 环境启动网页版。
> 内置依赖说明见 `pyproject.toml`（顶层依赖）与 `requirements-rag.txt`。

## 命令行示例

```bash
python 04_react_agent.py   # 多工具 ReAct Agent（终端交互）
python 07_混合检索评测.py   # 混合检索 vs 纯向量 对比
python 09_rerank评测.py    # Rerank 重排效果对比
python 06_复盘react.py     # 验证 Agent 是否为真 ReAct
python 05_ocr入库.py       # 图片 OCR 文字入库
```

## 技术栈

| 分类 | 技术 |
|------|------|
| **Agent 框架** | LangGraph（ReAct 状态图）· LangChain · `create_react_agent` |
| **LLM** | DeepSeek-chat（工具调用 / 结构化输出） |
| **向量检索** | BGE 中文嵌入 `bge-small-zh-v1.5` · Chroma 向量库 · sentence-transformers |
| **关键词检索** | BM25（自研中文切词） |
| **融合** | RRF（Reciprocal Rank Fusion） |
| **重排精排** | Cross-Encoder `bge-reranker-base` + 置信度阈值过滤 |
| **OCR** | RapidOCR · Pillow 字节流解码 |
| **后端** | Python 原生 HTTP Server（`http.server`，零第三方 Web 依赖） |
| **前端** | 原生 HTML/CSS/JS（蓝绿医疗主题、置信度来源展示、BYOK 输入） |
| **接口** | RESTful：`/api/chat`（Agent 对话）、`/api/ocr`（图片识别） |

## 项目结构

```
medical-rag/
├── 01_建知识库.py        # 构建向量知识库
├── 02_问答.py            # 固定流程 RAG 问答（对比 Agent）
├── 03_评测.py            # RAG 检索评测
├── 04_react_agent.py     # 多工具 ReAct Agent（终端交互）
├── 05_ocr入库.py         # 图片 OCR 文字入库
├── 06_复盘react.py       # 验证 Agent 是否为真 ReAct
├── 07_混合检索评测.py    # 混合检索 vs 纯向量 对比
├── 09_rerank评测.py      # Rerank 重排效果对比
├── agent_tools.py        # 多工具集（知识检索/紧急分级/剂量计算/单位换算）
├── hybrid_retriever.py   # 混合检索器（向量+BM25+RRF）
├── reranker.py           # Cross-Encoder 重排器
├── data/                 # 知识库源文档
└── web/                  # 网页版后端 + 前端
```