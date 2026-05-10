"""Rerank 精排 — 对召回结果用专用模型重新打分

流程: 召回 top20 → Rerank 精排 → 取 top5
模型: BAAI/bge-reranker-v2-m3 (SiliconFlow)
"""

import requests
from dataclasses import dataclass

from config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL

RERANK_URL = EMBEDDING_BASE_URL.rstrip("/").replace("/v1", "") + "/v1/rerank"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


@dataclass
class RerankResult:
    index: int
    score: float
    text: str


def rerank(query: str, documents: list[str], top_n: int = 5) -> list[RerankResult]:
    """对文档列表做精排，返回按相关性排序的结果"""
    if not documents:
        return []

    # 去掉空文档
    doc_map = [(i, doc) for i, doc in enumerate(documents) if doc.strip()]
    if not doc_map:
        return []

    try:
        resp = requests.post(
            RERANK_URL,
            headers={
                "Authorization": f"Bearer {EMBEDDING_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": [doc for _, doc in doc_map],
                "top_n": min(top_n, len(doc_map)),
                "return_documents": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        # rerank 失败时回退：按原始顺序返回前 top_n 条
        return [
            RerankResult(index=orig_idx, score=1.0 - i * 0.01, text=doc)
            for i, (orig_idx, doc) in enumerate(doc_map[:top_n])
        ]

    results = []
    for item in data.get("results", []):
        rerank_idx = item["index"]
        orig_idx, text = doc_map[rerank_idx]
        results.append(RerankResult(
            index=orig_idx,
            score=item["relevance_score"],
            text=text,
        ))

    return results[:top_n]
