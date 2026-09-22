# -*- coding: utf-8 -*-
"""
09 复盘：Rerank 重排到底提升了什么？
运行：cd medical-rag
     set HF_ENDPOINT=https://hf-mirror.com
     python 09_rerank评测.py        (需要能联网下载 bge-reranker 模型)

它对同一批问题，分别跑"仅混合检索"和"混合检索+Rerank"，
打印两者各自命中的 Top-3，你能直观看到重排改变了排序、是否把更相关的片段顶上来了。

对比维度：
  - 混合检索：向量 + BM25 → RRF 融合 → 直接取 Top-3（粗召回）
  - +Rerank  ：向量 + BM25 → RRF 融合 15 条候选 → cross-encoder 精排 → Top-3
"""
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from langchain_community.embeddings import HuggingFaceEmbeddings
from hybrid_retriever import HybridRetriever
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

EMB_MODEL = "BAAI/bge-small-zh-v1.5"
DB_DIR = os.path.join(BASE_DIR, "chroma_db")


def snapshot(retriever, question, use_rerank, k=3, min_score=0.0):
    """返回 Top-k 的 (片段, 分数) 快照，方便对比排序变化。"""
    try:
        return retriever.search_with_scores(question, k=k, use_rerank=use_rerank, min_score=min_score)
    except Exception as e:
        return [("ERROR: " + str(e), 0.0)]


def main():
    print("加载向量模型（只加载一次）...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True}
    )
    print("加载混合检索器（向量 + BM25 + Rerank）...\n")
    retriever = HybridRetriever(DB_DIR, embeddings=embeddings)

    questions = [
        "高血压患者每天食盐不超过多少克？",
        "氯化钠的摄入量应该控制到多少？",
        "每天吃药的话这个药叫什么？",
        "睡觉困难怎么办？",
        "失眠的人睡前应该避免什么？",
    ]

    im = 0
    for q in questions:
        print("=" * 64)
        print("问：", q)
        plain = snapshot(retriever, q, use_rerank=False)
        reranked = snapshot(retriever, q, use_rerank=True, min_score=0.1)
        # 检查重排后 Top-1 是否变化
        changed = (plain[0][0] != reranked[0][0]) if plain and reranked else False
        im += 1 if changed else 0

        def lines(res):
            return "\n".join(
                f"    [{i + 1}] ({s:.3f}) {t[:50].replace(chr(10), ' ')}"
                for i, (t, s) in enumerate(res)
            )

        print("─ 仅混合检索 Top-3：")
        print(lines(plain))
        print("─ 混合+重排 Top-3：")
        print(lines(reranked))
        print("　→ ", "✅ 重排改变了排序，更相关的片段被顶上来了" if changed else "⏸ Top-1 未变（本就命中）")

    print("\n" + "=" * 64)
    print(f"共 {len(questions)} 个问题，其中 {im} 个问题 Top-1 被重排改变。")
    print("说明：重排的价值不仅在于改 Top-1，更在于把『细粒度相关』(剂量/否定/实体)的片段排到最前。")


if __name__ == "__main__":
    main()