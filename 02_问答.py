"""
02 RAG 问答（完整闭环：检索 → 增强 → 生成）

相比 01 的"直接问模型"，这里多了一个核心动作：先到知识库里检索最相关的资料，
再把资料和问题一起交给 DeepSeek，让它"有据可查"地回答。

运行：python 02_问答.py
"""
import os
from dotenv import load_dotenv

load_dotenv()  # 复用 dantong-agent 的 .env 思路（medical-rag 下也可自建）

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")


def get_retriever():
    """1. 打开已建好的向量库，做成一个"检索器"。"""
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
    db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )
    return db.as_retriever(search_kwargs={"k": 3})  # 每次取最相关的 3 条


def get_llm():
    """2. 创建 DeepSeek 大模型（和 SOP 项目一样的用法）。"""
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
        temperature=0,  # 医疗回答要严谨，用 0
    )


def ask(question: str):
    retriever = get_retriever()
    llm = get_llm()

    # ---- 核心：检索 ----
    print(f"🔍 检索与问题最相关的资料...")
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)

    # ---- 核心：增强（把资料拼进提示词） + 生成 ----
    # 提示词说明"只依据给出资料回答，资料没有的要说不知道"（防幻觉）
    prompt = f"""你是专业的健康顾问。请只依据下面提供的资料回答用户问题。
如果资料里没有相关信息，请明确说"资料中未提及"，不要自行编造。

【资料】
{context}

【问题】
{question}

请给出简洁、准确的回答。"""
    print("🤖 基于检索资料组织回答...")
    resp = llm.invoke(prompt)

    print("\n" + "=" * 50)
    print("📌 回答：")
    print(resp.content)

    # ---- 展示引用的资料出处（体现"有据可查"）----
    print("\n📚 本回答依据的 3 条知识片段：")
    for i, d in enumerate(docs, 1):
        print(f"  --- 片段 {i} ---")
        print(d.page_content)


if __name__ == "__main__":
    q = "高血压饮食要注意什么？"
    ask(q)