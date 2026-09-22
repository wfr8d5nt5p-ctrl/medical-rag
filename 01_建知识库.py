"""
01 建知识库（RAG 的第一步：准备"开卷考试的参考书"）

作用：把 data/ 里的健康资料，切成小块 → 转成向量 → 存进 Chroma 向量库。
下次生成答案前，就可以从这里面"检索"最相关的段落。

RAG 一句话版：
  先把你手上的资料准备好（本脚本），再在提问时检索+回答（见 02_问答.py）。

运行：python 01_建知识库.py
首次会下载中文向量模型（几十~几百 MB），之后走本地缓存，会快很多。
"""
import os

# 让代码在运行本脚本时，即使从别的目录调用也能找到 data 和 db
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader


def load_docs():
    """1. 读取 data 目录下所有 .md 文档。"""
    docs = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith((".md", ".txt")):
            loader = TextLoader(os.path.join(DATA_DIR, fname), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def build_embeddings():
    """
    2. 创建"中文向量化模型"。
    作用：把一段文字变成一个数字数组（向量），让相似句子向量相近。
    用 BAAI 的中文轻量模型 bge-small-zh，体积小、效果不错，适合入门。
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )


def split_docs(docs):
    """3. 把长文章切成小块（chunk），每块约 200 字、重叠 50 字。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,      # 每块最大字符数
        chunk_overlap=50,    # 相邻块重叠，避免语义被切断
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    return splitter.split_documents(docs)


def main():
    print("📄 读取资料...")
    docs = load_docs()
    print(f"   共读取 {len(docs)} 个文档")

    print("✂️  切块...")
    chunks = split_docs(docs)
    print(f"   切成 {len(chunks)} 个小块")

    print("🧠 初始化中文向量模型（首次会下载，请耐心等待）...")
    embeddings = build_embeddings()

    print("💾 写入 Chroma 向量库...")
    # persist_directory 指定保存位置，'a' 表示追加/重建
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR,
    )
    db.persist()
    print(f"✅ 知识库已建好，保存到：\n   {DB_DIR}")
    print(f"   共入库 {db._collection.count()} 个片段。下一步运行 02_问答.py 试试提问。")


if __name__ == "__main__":
    main()