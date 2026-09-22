# -*- coding: utf-8 -*-
"""
07 混合检索评测：纯向量 vs 向量+BM25（RRF 融合）

运行（medical-rag 目录下）：
    set HF_ENDPOINT=https://hf-mirror.com
    python 07_混合检索评测.py

只依赖向量模型（本地缓存），不调 LLM，跑得很快。
输出：每题两种检索的 Top3 是否命中期望关键词，最后给出召回率对比。
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")
sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from hybrid_retriever import HybridRetriever

EMB_MODEL = "BAAI/bge-small-zh-v1.5"

# (问题, 期望命中的关键词列表) —— 关键词取自知识库原文
TESTS = [
    # ---- 基础题（同义直问，两者都应命中）----
    ("高血压患者每天食盐不超过多少克？", ["5 克", "5克", "5 克盐"]),
    ("失眠的人睡前应该避免什么？", ["电子屏幕", "咖啡", "手机", "浓茶"]),
    ("张三的化验单诊断是什么病？", ["高血压二级"]),
    ("体检前一天饮食上要注意什么？", ["清淡", "饮酒", "剧烈运动"]),
    ("高血压患者应减少哪种物质的摄入？", ["钠盐", "钠", "盐"]),
    # ---- 高区分度题（考纯向量盲区：同义词/术语/字面/倒装）----
    ("氯化钠的摄入量应该控制到多少？", ["5 克", "5克", "钠盐"]),          # 同义词：氯化钠 → 钠盐
    ("每天吃药的话这个药叫什么？", ["硝苯地平"]),                          # 泛指 → 具体药名
    ("睡觉困难怎么办？", ["失眠", "规律作息", "电子屏幕"]),                # 口语 → 术语
    ("查一下张三的单子上写没写复查时间", ["一个月", "复诊"]),              # 口语化 → 原文
    ("体检前一天晚上喝啤酒行不行？", ["饮酒", "酒精"]),                    # 倒装/口语 → 原文
]


def pure_vector_search(question: str, db, k: int = 3):
    return [d.page_content for d in db.similarity_search(question, k=k)]


def hit(texts, keywords):
    blob = "".join(texts)
    return any(kw in blob for kw in keywords)


def main():
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("加载向量模型（只加载一次）...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    print("加载混合检索器（含 BM25 索引）...")
    hybrid = HybridRetriever(DB_DIR, embeddings=embeddings)
    print("加载纯向量检索器（复用同一模型）...")
    pure_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("\n" + "=" * 64)
    print(f"{'问题':<34}{'纯向量':<8}{'混合':<8}")
    print("-" * 64)

    pure_hit = 0
    hybrid_hit = 0
    for i, (q, kws) in enumerate(TESTS, 1):
        p_texts = pure_vector_search(q, pure_db)
        h_texts = hybrid.search(q, k=3)
        p_ok = hit(p_texts, kws)
        h_ok = hit(h_texts, kws)
        pure_hit += p_ok
        hybrid_hit += h_ok
        p_mark = "✅" if p_ok else "❌"
        h_mark = "✅" if h_ok else "❌"
        print(f"{i}. {q[:32]:<30}  {p_mark:^6}  {h_mark:^6}")

    n = len(TESTS)
    print("-" * 64)
    print(f"召回率：纯向量 {pure_hit}/{n} ({pure_hit / n:.0%})   "
          f"混合 {hybrid_hit}/{n} ({hybrid_hit / n:.0%})")
    print("=" * 64)

    # 列出差异题，便于分析
    print("\n差异明细（混合与纯向量命中不一致的题）：")
    any_diff = False
    for i, (q, kws) in enumerate(TESTS, 1):
        p_ok = hit(pure_vector_search(q, pure_db), kws)
        h_ok = hit(hybrid.search(q, k=3), kws)
        if p_ok != h_ok:
            any_diff = True
            print(f"  第{i}题 [{q}]  纯向量={'命中' if p_ok else '未命中'} → 混合={'命中' if h_ok else '未命中'}")
            if not h_ok and p_ok:
                print("       ↑ 混合反而更差，可调 vector_top/bm25_top 或 k 再测")
    if not any_diff:
        print("  无差异（两组命中情况一致）")


if __name__ == "__main__":
    main()
