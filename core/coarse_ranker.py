"""粗排 — 融合向量召回和 BM25 关键词召回，快速过滤

流程:
  向量召回 top30 + BM25召回 top30 → 去重 → RRF融合打分 → 取 top15 → 送精排

RRF (Reciprocal Rank Fusion): 每条结果的分数 = Σ 1/(k + rank_i)
  多路召回的标准融合算法，不需要归一化，对排名差异鲁棒
"""

from __future__ import annotations

import math
import jieba
from dataclasses import dataclass, field

RRF_K = 60  # RRF 常数，越大越平滑


@dataclass
class CoarseResult:
    text: str
    score: float
    sources: list[str] = field(default_factory=list)  # 来自哪路召回
    metadata: dict = field(default_factory=dict)


def coarse_rank(
    query: str,
    vector_results: list[dict],
    bm25_corpus: list[dict] | None = None,
    top_n: int = 15,
) -> list[CoarseResult]:
    """粗排：融合向量召回 + BM25 召回

    Args:
        query: 用户查询
        vector_results: 向量召回结果 [{"text": str, "score": float, "metadata": dict}, ...]
        bm25_corpus: BM25 语料（None 则只用向量结果做粗排）
        top_n: 粗排后保留条数
    """
    scored: dict[str, CoarseResult] = {}

    # 向量召回 RRF
    for rank, item in enumerate(vector_results):
        text = item["text"]
        rrf = 1.0 / (RRF_K + rank + 1)
        if text in scored:
            scored[text].score += rrf
            scored[text].sources.append("vector")
        else:
            scored[text] = CoarseResult(
                text=text, score=rrf, sources=["vector"], metadata=item.get("metadata", {}),
            )

    # BM25 召回 RRF
    if bm25_corpus:
        bm25_results = _bm25_search(query, bm25_corpus, top_n=top_n * 2)
        for rank, item in enumerate(bm25_results):
            text = item["text"]
            rrf = 1.0 / (RRF_K + rank + 1)
            if text in scored:
                scored[text].score += rrf
                if "bm25" not in scored[text].sources:
                    scored[text].sources.append("bm25")
            else:
                scored[text] = CoarseResult(
                    text=text, score=rrf, sources=["bm25"], metadata=item.get("metadata", {}),
                )

    # 两路都命中的结果得分自然更高（RRF 累加）
    ranked = sorted(scored.values(), key=lambda x: x.score, reverse=True)
    return ranked[:top_n]


# ── BM25 实现 ──

def _bm25_search(query: str, corpus: list[dict], top_n: int = 30) -> list[dict]:
    """轻量 BM25 关键词搜索"""
    query_terms = list(jieba.cut_for_search(query))
    if not query_terms or not corpus:
        return []

    texts = [item["text"] for item in corpus]
    doc_count = len(texts)

    # 分词
    doc_terms = [list(jieba.cut_for_search(t)) for t in texts]
    doc_lens = [len(dt) for dt in doc_terms]
    avg_dl = sum(doc_lens) / doc_count if doc_count else 1

    # IDF
    df = {}
    for dt in doc_terms:
        seen = set(dt)
        for term in seen:
            df[term] = df.get(term, 0) + 1

    idf = {}
    for term in query_terms:
        n = df.get(term, 0)
        idf[term] = math.log((doc_count - n + 0.5) / (n + 0.5) + 1)

    # BM25 打分
    k1, b = 1.5, 0.75
    scores = []
    for i, dt in enumerate(doc_terms):
        tf_map = {}
        for term in dt:
            tf_map[term] = tf_map.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_lens[i] / avg_dl)
            score += idf.get(term, 0) * numerator / denominator

        scores.append((score, i))

    scores.sort(reverse=True)

    return [
        {**corpus[idx], "bm25_score": s}
        for s, idx in scores[:top_n]
        if s > 0
    ]
