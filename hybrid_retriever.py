# -*- coding: utf-8 -*-
"""
混合检索器：向量检索（语义） + BM25（字面） → RRF 融合

为什么混合：
  纯向量检索擅长"语义相近"，但专有名词/剂量/缩写等字面匹配容易漏召回；
  BM25 按词频+稀有度打分，字面命中稳，但不懂语义改写。
  两者取 Top-N，再用 RRF（Reciprocal Rank Fusion）融合排名，取最终 Top-k。

零第三方依赖：BM25 自实现（中文按 单字+相邻二元组 切词，英文/数字按整词）。
"""
import math
import re

CJK_RE = re.compile(r"[\u4e00-\u9fff]")
WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> list:
    """中文：单字 + 相邻二元组；英文/数字：整词（小写）。"""
    tokens = []
    for w in WORD_RE.findall(text):
        tokens.append(w.lower())
    chars = CJK_RE.findall(text)
    tokens.extend(chars)
    for i in range(len(chars) - 1):
        tokens.append(chars[i] + chars[i + 1])
    return tokens


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = [tokenize(doc) for doc in corpus]
        self.dl = [len(t) for t in self.corpus]
        self.avgdl = sum(self.dl) / len(self.dl) if self.dl else 1.0
        self.N = len(self.corpus)
        self.df = {}
        for toks in self.corpus:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.k1, self.b = k1, b

    def score(self, query: str):
        q_tokens = set(tokenize(query))
        scores = [0.0] * self.N
        for t in q_tokens:
            df = self.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
            for i, toks in enumerate(self.corpus):
                tf = toks.count(t)
                if tf == 0:
                    continue
                dl = self.dl[i]
                scores[i] += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                )
        return scores


def rrf_fuse(ranked_lists, k=60):
    """RRF 融合：score = Σ 1/(k + rank)。k 取 60 是论文常用值。"""
    acc = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            acc[doc_id] = acc.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(acc.items(), key=lambda x: -x[1])


class HybridRetriever:
    def __init__(self, db_dir, model_name="BAAI/bge-small-zh-v1.5", embeddings=None):
        from langchain_community.vectorstores import Chroma

        if embeddings is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
        self.embeddings = embeddings
        self.db = Chroma(persist_directory=db_dir, embedding_function=self.embeddings)
        raw = self.db.get()
        self.chunks = raw.get("documents", [])
        self.content2idx = {t: i for i, t in enumerate(self.chunks)}
        self.bm25 = BM25(self.chunks)

    def _vec_ranked(self, question, top):
        docs = self.db.similarity_search(question, k=top)
        ranked = []
        for d in docs:
            idx = self.content2idx.get(d.page_content)
            if idx is not None and idx not in ranked:
                ranked.append(idx)
        return ranked

    def _bm25_ranked(self, question, top):
        scores = self.bm25.score(question)
        return sorted(range(len(scores)), key=lambda i: -scores[i])[:top]

    def search_with_scores(self, question, k=3, vector_top=30, bm25_top=30, use_rerank=True, cand=15, rerank_k=None, min_score=0.0):
        """
        混合检索 + Rerank：返回 [(text, score)]。
        流程：向量 + BM25 各取 Top-30 → RRF 融合出候选 cand=15
              → cross-encoder 重排器精排 → 取最终 Top-k
        score：重排后返回重排器分数(高=更相关)；重排不可用时回落到向量相关度。
        min_score：只保留重排分数 >= min_score 的片段（用于剔除无关数据，宁缺毋滥）。
                   设为 0.0 表示不截断（默认）。
        """
        fused = rrf_fuse(
            [self._vec_ranked(question, vector_top), self._bm25_ranked(question, bm25_top)]
        )
        cand_ids = [idx for idx, _ in fused[:cand]]
        cand_texts = [self.chunks[idx] for idx in cand_ids]

        # 原始向量相关度(供降级时展示)
        try:
            scored = self.db.similarity_search_with_relevance_scores(question, k=cand + 5)
            score_map = {d.page_content: float(s) for d, s in scored}
        except Exception:
            score_map = {}

        if use_rerank and cand_texts:
            from reranker import rerank
            rr = rerank(question, cand_texts, top_k=rerank_k or k, score_map=score_map)
            # 阈值截断：剔除无关片段（分数低于 min_score 的不返回）
            if min_score > 0:
                rr = [(t, s) for t, s in rr if s >= min_score]
            return [(text, round(float(score), 4)) for text, score in rr]

        # 无重排：返回融合结果，分数用向量相关度
        out = []
        for idx in cand_ids[:k]:
            text = self.chunks[idx]
            out.append((text, round(score_map.get(text, 0.0), 3)))
        return out

    def search(self, question, k=3, min_score=0.0):
        """混合检索 + Rerank：只返回命中的文本列表（可带阈值过滤）。"""
        return [text for text, _ in self.search_with_scores(question, k=k, min_score=min_score)]
