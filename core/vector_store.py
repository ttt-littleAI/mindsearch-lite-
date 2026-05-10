"""Milvus 多集合向量存储 — document_chunks / search_cache / user_memory

每个集合职责:
  document_chunks: 用户上传的文档（PDF/Word/图片等解析后的分块）
  search_cache:    搜索结果缓存（Searcher产出的摘要，避免重复搜索）
  user_memory:     用户长期记忆（偏好、常用领域等，Phase 6 启用）
"""

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pymilvus import MilvusClient, DataType

from config import (
    MILVUS_URI,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
)

COLLECTIONS = {
    "document_chunks": {
        "description": "用户上传文档的子块（向量检索用）；parent_id 关联 MySQL parent_chunks 表",
        "fields": [
            ("id", DataType.INT64, {"is_primary": True, "auto_id": True}),
            ("embedding", DataType.FLOAT_VECTOR, {"dim": EMBEDDING_DIM}),
            ("text", DataType.VARCHAR, {"max_length": 65535}),
            ("source", DataType.VARCHAR, {"max_length": 2048}),
            ("file_type", DataType.VARCHAR, {"max_length": 32}),
            ("page_number", DataType.INT32, {}),
            ("user_id", DataType.VARCHAR, {"max_length": 128}),
            ("parent_id", DataType.VARCHAR, {"max_length": 64}),  # 空字符串表示无父块（兼容旧数据）
        ],
    },
    "search_cache": {
        "description": "搜索结果缓存",
        "fields": [
            ("id", DataType.INT64, {"is_primary": True, "auto_id": True}),
            ("embedding", DataType.FLOAT_VECTOR, {"dim": EMBEDDING_DIM}),
            ("text", DataType.VARCHAR, {"max_length": 65535}),
            ("source", DataType.VARCHAR, {"max_length": 2048}),
            ("query", DataType.VARCHAR, {"max_length": 2048}),
        ],
    },
    "user_memory": {
        "description": "用户长期记忆",
        "fields": [
            ("id", DataType.INT64, {"is_primary": True, "auto_id": True}),
            ("embedding", DataType.FLOAT_VECTOR, {"dim": EMBEDDING_DIM}),
            ("text", DataType.VARCHAR, {"max_length": 65535}),
            ("memory_type", DataType.VARCHAR, {"max_length": 64}),
            ("user_id", DataType.VARCHAR, {"max_length": 128}),
        ],
    },
}


class VectorStore:
    """Milvus 多集合管理器"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
            model=EMBEDDING_MODEL,
        )
        self.client = MilvusClient(uri=MILVUS_URI)

    def ensure_collections(self):
        for name, config in COLLECTIONS.items():
            if self.client.has_collection(name):
                continue
            schema = self.client.create_schema()
            for field_name, dtype, params in config["fields"]:
                schema.add_field(field_name, dtype, **params)

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="IVF_FLAT",
                params={"nlist": 128},
            )
            self.client.create_collection(name, schema=schema, index_params=index_params)

    # ── document_chunks ──

    def add_document_chunks(self, chunks: list, user_id: str = "default"):
        """存入文档分块（接受 core.chunker.Chunk 列表）。

        如果 chunk.metadata 包含 parent_id，会一并写入 Milvus；否则 parent_id 为空字符串。
        """
        if not chunks:
            return
        texts = [c.text for c in chunks]
        vectors = self.embeddings.embed_documents(texts)
        data = [
            {
                "embedding": vec,
                "text": c.text,
                "source": c.source_file,
                "file_type": c.metadata.get("file_type", ""),
                "page_number": c.page_number if c.page_number is not None else -1,
                "user_id": user_id,
                "parent_id": c.metadata.get("parent_id", ""),
            }
            for vec, c in zip(vectors, chunks)
        ]
        self.client.insert("document_chunks", data)
        self.client.flush("document_chunks")

    def add_document_hierarchical(self, parents: list, children: list, user_id: str = "default"):
        """父子切分入库：父块全文存 MySQL，子块向量化存 Milvus（带 parent_id）。

        参数:
          parents:  list[chunker.ParentChunkData]
          children: list[chunker.Chunk]，每条 metadata 必须含 parent_id
        """
        from core.database import save_parent_chunks
        if not children:
            return
        # 1. 父块入 MySQL
        if parents:
            save_parent_chunks([
                {
                    "parent_id": p.parent_id,
                    "user_id": user_id,
                    "source_file": p.source_file,
                    "file_type": p.file_type,
                    "page_number": p.page_number if p.page_number is not None else -1,
                    "text": p.text,
                }
                for p in parents
            ])
        # 2. 子块入 Milvus（复用 add_document_chunks，metadata 已带 parent_id）
        self.add_document_chunks(children, user_id=user_id)

    def search_documents(self, query: str, k: int = 5, user_id: str | None = None) -> list[Document]:
        """父子检索: 向量召回子块 → 粗排+精排 → 去重 parent_id → 回查父块全文返回。

        三种情况:
          1. 子块有 parent_id → 用父块全文作为 page_content（父子模式）
          2. 子块无 parent_id（旧数据/直接调 add_document_chunks）→ 用子块文本兜底
          3. parent_id 在 MySQL 查不到（数据被删但 Milvus 残留）→ 用子块文本兜底
        """
        from core.reranker import rerank
        from core.coarse_ranker import coarse_rank
        from core.database import get_parent_chunks

        stats = self.client.get_collection_stats("document_chunks")
        if stats["row_count"] == 0:
            return []

        recall_k = min(30, max(k * 6, 20))
        query_vector = self.embeddings.embed_query(query)
        filter_expr = f'user_id == "{user_id}"' if user_id else None
        results = self.client.search(
            "document_chunks",
            data=[query_vector],
            limit=recall_k,
            output_fields=["text", "source", "file_type", "page_number", "parent_id"],
            filter=filter_expr,
        )

        if not results[0]:
            return []

        hits = results[0]

        # Stage 2: 粗排 — 向量排名 + BM25 关键词排名 RRF 融合（子块文本上做）
        vector_results = [
            {"text": hit["entity"]["text"], "score": hit["distance"]}
            for hit in hits
        ]
        bm25_corpus = [{"text": hit["entity"]["text"]} for hit in hits]
        coarse_results = coarse_rank(
            query, vector_results, bm25_corpus=bm25_corpus,
            top_n=min(15, len(hits)),
        )

        # Stage 3: 精排 — Rerank 模型打分（仍在子块文本上）
        coarse_texts = [cr.text for cr in coarse_results]
        # 多召回一些以补偿后续按 parent 去重导致的损失
        reranked = rerank(query, coarse_texts, top_n=k * 3)

        text_to_hit = {hit["entity"]["text"]: hit for hit in hits}

        # Stage 4: 父块替换 — 收集 parent_ids 一次性查 MySQL，用父块全文替代子块作为返回内容
        parent_ids = []
        for r in reranked:
            if r.text not in text_to_hit:
                continue
            pid = text_to_hit[r.text]["entity"].get("parent_id", "")
            if pid:
                parent_ids.append(pid)
        parent_map = get_parent_chunks(list(set(parent_ids))) if parent_ids else {}

        # Stage 5: 按 parent_id 去重（同一父块多个子命中只保留最高分），返回 top_k 父块
        seen_parents = set()
        final_docs = []
        for r in reranked:
            if r.text not in text_to_hit:
                continue
            hit = text_to_hit[r.text]
            pid = hit["entity"].get("parent_id", "")
            if pid and pid in parent_map:
                if pid in seen_parents:
                    continue
                seen_parents.add(pid)
                parent = parent_map[pid]
                final_docs.append(Document(
                    page_content=parent["text"],
                    metadata={
                        "source": parent["source_file"],
                        "file_type": parent["file_type"],
                        "page_number": parent["page_number"],
                        "score": r.score,
                        "parent_id": pid,
                        "matched_child": r.text,  # 命中的子块文本（debug 用）
                    },
                ))
            else:
                # 兼容：无 parent_id 或父块缺失，回退到子块文本
                final_docs.append(Document(
                    page_content=r.text,
                    metadata={
                        "source": hit["entity"]["source"],
                        "file_type": hit["entity"]["file_type"],
                        "page_number": hit["entity"]["page_number"],
                        "score": r.score,
                    },
                ))
            if len(final_docs) >= k:
                break

        return final_docs

    # ── search_cache ──

    def cache_search_result(self, query: str, text: str, source: str = ""):
        vector = self.embeddings.embed_documents([text])[0]
        self.client.insert("search_cache", [{
            "embedding": vector,
            "text": text,
            "source": source,
            "query": query[:2048],
        }])
        self.client.flush("search_cache")

    def search_cache(self, query: str, k: int = 3, threshold: float = 0.85) -> list[Document]:
        stats = self.client.get_collection_stats("search_cache")
        if stats["row_count"] == 0:
            return []
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            "search_cache",
            data=[query_vector],
            limit=k,
            output_fields=["text", "source", "query"],
        )
        return [
            Document(
                page_content=hit["entity"]["text"],
                metadata={
                    "source": hit["entity"]["source"],
                    "original_query": hit["entity"]["query"],
                    "score": hit["distance"],
                },
            )
            for hit in results[0]
            if hit["distance"] >= threshold
        ]

    # ── user_memory ──

    def add_user_memory(self, text: str, memory_type: str, user_id: str):
        vector = self.embeddings.embed_documents([text])[0]
        self.client.insert("user_memory", [{
            "embedding": vector,
            "text": text,
            "memory_type": memory_type,
            "user_id": user_id,
        }])
        self.client.flush("user_memory")

    def search_user_memory(self, query: str, user_id: str, k: int = 3) -> list[Document]:
        stats = self.client.get_collection_stats("user_memory")
        if stats["row_count"] == 0:
            return []
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            "user_memory",
            data=[query_vector],
            limit=k,
            output_fields=["text", "memory_type"],
            filter=f'user_id == "{user_id}"',
        )
        return [
            Document(
                page_content=hit["entity"]["text"],
                metadata={
                    "memory_type": hit["entity"]["memory_type"],
                    "score": hit["distance"],
                },
            )
            for hit in results[0]
        ]

    # ── 兼容旧接口 ──

    def add_documents(self, docs: list[Document]):
        """兼容旧 SearchMemory.add_documents"""
        if not docs:
            return
        texts = [doc.page_content for doc in docs]
        vectors = self.embeddings.embed_documents(texts)
        data = [
            {
                "embedding": vec,
                "text": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "query": doc.metadata.get("query", ""),
            }
            for vec, doc in zip(vectors, docs)
        ]
        self.client.insert("search_cache", data)
        self.client.flush("search_cache")

    def search(self, query: str, k: int = 3) -> list[Document]:
        """召回 → 粗排(BM25+RRF) → 精排(Rerank) — 搜索 search_cache + document_chunks"""
        from core.reranker import rerank
        from core.coarse_ranker import coarse_rank

        cache_results = self.search_cache(query, k=k * 5, threshold=0.0)
        doc_results = self._search_documents_raw(query, k=k * 5)
        combined = cache_results + doc_results

        if not combined:
            return []

        # Stage 2: 粗排
        vector_results = [
            {"text": doc.page_content, "score": doc.metadata.get("score", 0)}
            for doc in combined
        ]
        bm25_corpus = [{"text": doc.page_content} for doc in combined]
        coarse_results = coarse_rank(
            query, vector_results, bm25_corpus=bm25_corpus,
            top_n=min(15, len(combined)),
        )

        # Stage 3: 精排
        coarse_texts = [cr.text for cr in coarse_results]
        reranked = rerank(query, coarse_texts, top_n=k)

        text_to_idx = {}
        for i, doc in enumerate(combined):
            if doc.page_content not in text_to_idx:
                text_to_idx[doc.page_content] = i

        return [
            Document(
                page_content=r.text,
                metadata={**combined[text_to_idx[r.text]].metadata, "score": r.score},
            )
            for r in reranked
            if r.text in text_to_idx
        ]

    def _search_documents_raw(self, query: str, k: int = 10) -> list[Document]:
        """文档召回（不含 rerank，内部使用）"""
        stats = self.client.get_collection_stats("document_chunks")
        if stats["row_count"] == 0:
            return []
        query_vector = self.embeddings.embed_query(query)
        results = self.client.search(
            "document_chunks",
            data=[query_vector],
            limit=k,
            output_fields=["text", "source", "file_type", "page_number"],
        )
        return [
            Document(
                page_content=hit["entity"]["text"],
                metadata={
                    "source": hit["entity"]["source"],
                    "file_type": hit["entity"]["file_type"],
                    "page_number": hit["entity"]["page_number"],
                    "score": hit["distance"],
                },
            )
            for hit in results[0]
        ]


_store_instance: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
        _store_instance.ensure_collections()
    return _store_instance
