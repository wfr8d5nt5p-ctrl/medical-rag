# -*- coding: utf-8 -*-
"""
Rerank 重排器：用 cross-encoder（BGE Reranker）对"候选片段"二次打分排序。

为什么要重排：
  混合检索（向量+BM25）产出的 Top-N 是"粗召回"，分数只能反映"大致相关"，
  而 cross-encoder 会把【查询】与【片段】拼接后整段进模型计算"精确匹配"，
  对专有名词、剂量、否定表述(如"不要吃")、实体冲突做细粒度判断，
  显著提升 Top-1 / Top-3 的相关度（简历里"重排检索"的核心）。

与普通 embedding 的区别：
  - bi-encoder(embeddings)：把 query 和 doc 各自编码成一个向量，算余弦相似度（快，可离线，但精度粗）。
  - cross-encoder(reranker)：query 和 doc 拼一起过模型，直接输出相关性分数（慢，但精度高）。

因此正确的用法是"先粗召回多数，再精排少数"。
"""
import os

_RERANKER = None


def get_reranker(model_name="BAAI/bge-reranker-base"):
    """懒加载重排模型(SentenceTransformers CrossEncoder)，只加载一次。"""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder

        _RERANKER = CrossEncoder(model_name_or_path=model_name, max_length=512)
    return _RERANKER


def rerank(question, candidates, top_k=3, score_map=None):
    """
    对候选片段重排，返回 [(text, rerank_score)]。
    candidates: list[str] 已粗召回的文本。
    score_map : dict 可选，附带原始相关度(如向量分)，返回时合并进去展示。
    若模型加载失败(缺依赖/无网络/下载失败)，优雅降级为原顺序，保证服务不崩。
    """
    if not candidates:
        return []
    # 先去重：知识库里可能存有内容完全相同的 chunk，避免重排后重复出现在 Top-k
    seen, unique = set(), []
    for doc in candidates:
        if doc not in seen:
            seen.add(doc)
            unique.append(doc)
    # 有序去重后若不足 top_k，需保证仍有候选返回
    try:
        model = get_reranker()
        pairs = [[question, doc] for doc in unique]
        scores = model.predict(pairs)
        ranked = sorted(
            zip(unique, scores), key=lambda x: -x[1]
        )[:top_k]
    except Exception as e:
        # 降级：模型不可用就用原顺序，保留前 top_k 条
        print(f"[Rerank] 模型不可用，已降级为粗召回顺序：{e}")
        nodegrade = [(doc, 0.0) for doc in unique[:top_k]]
        if score_map:
            nodegrade = [(doc, float(score_map.get(doc, 0.0))) for doc in unique[:top_k]]
        ranked = nodegrade
    return ranked