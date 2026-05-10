"""MindSearch MCP Server — 将搜索引擎能力暴露为标准 MCP 工具

工具列表:
  smart_search      — 智能搜索（自动意图路由）
  web_search        — 网络实时搜索
  knowledge_search  — 本地知识库检索（三级精排管线）
  document_parse    — 文档解析与入库
  memory_recall     — 用户记忆召回
  hot_queries       — 热门搜索排行

启动: python mcp_server.py
配置: Claude Code / Cursor 等客户端通过 stdio 连接
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="mindsearch",
    version="2.0.0",
)


@mcp.tool()
def smart_search(question: str, user_id: str = "default") -> str:
    """智能搜索：LLM 自动识别查询意图（REALTIME/STABLE/PERSONAL），
    路由到最优搜索策略（web_only/hybrid/local_only），返回带引用的回答。

    Args:
        question: 搜索问题，例如 "Transformer的自注意力机制是如何工作的"
        user_id: 用户ID，用于个性化搜索和记忆
    """
    from core.search_engine import search, SearchRequest, SearchMode
    result = search(SearchRequest(
        question=question,
        search_mode=SearchMode.AUTO,
        user_id=user_id,
    ))
    return json.dumps({
        "answer": result.answer,
        "route": result.metadata.get("route", ""),
        "strategy": result.metadata.get("strategy", ""),
        "cache": result.metadata.get("cache", ""),
        "duration_ms": result.metadata.get("duration_ms", 0),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def web_search(question: str) -> str:
    """网络实时搜索：直接搜索互联网获取最新信息。
    适用于新闻、热点事件、实时数据等需要最新信息的问题。

    Args:
        question: 搜索问题，例如 "2025年最新的AI政策"
    """
    from core.search_engine import search, SearchRequest, SearchMode
    result = search(SearchRequest(
        question=question,
        search_mode=SearchMode.WEB,
    ))
    return json.dumps({
        "answer": result.answer,
        "metadata": result.metadata,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def knowledge_search(question: str, user_id: str = "default", top_k: int = 5) -> str:
    """本地知识库检索：从用户上传的文档中检索，经过三级精排管线
    （向量召回 → BM25+RRF粗排 → Rerank精排）。不联网，隐私安全。

    Args:
        question: 检索问题
        user_id: 用户ID
        top_k: 返回条数，默认5
    """
    from core.vector_store import get_vector_store
    store = get_vector_store()
    docs = store.search_documents(question, k=top_k, user_id=user_id)

    if not docs:
        return json.dumps({"status": "empty", "message": "未找到相关文档，请先上传文档。"}, ensure_ascii=False)

    return json.dumps({
        "status": "found",
        "count": len(docs),
        "results": [
            {
                "content": doc.page_content[:500],
                "source": doc.metadata.get("source", ""),
                "score": round(doc.metadata.get("score", 0), 4),
            }
            for doc in docs
        ],
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def document_parse(file_path: str, user_id: str = "default") -> str:
    """文档解析：解析文档（PDF/Word/PPT/Excel）并自动分块存入向量库。
    支持 MinerU 深度解析，包括表格、图片 OCR。

    Args:
        file_path: 文档文件的绝对路径
        user_id: 用户ID
    """
    import asyncio
    from tools.document_parser import parse_document
    from core.chunker import chunk_document_chunks_hierarchical
    from core.vector_store import get_vector_store

    doc_chunks = asyncio.run(parse_document(file_path))
    parents, chunks = chunk_document_chunks_hierarchical(doc_chunks)

    store = get_vector_store()
    store.add_document_hierarchical(parents, chunks, user_id)

    return json.dumps({
        "file": os.path.basename(file_path),
        "parent_chunks": len(parents),
        "child_chunks": len(chunks),
        "status": "parsed_and_stored",
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def memory_recall(query: str, user_id: str) -> str:
    """记忆召回：查询用户的三层记忆（短期对话历史 + 长期偏好 + 工作记忆）。

    Args:
        query: 召回关键词
        user_id: 用户ID
    """
    from core.memory import get_ai_memory
    memory = get_ai_memory(user_id)
    recalled = memory.recall_all(query)
    return json.dumps(recalled, ensure_ascii=False, indent=2)


@mcp.tool()
def hot_queries(top_n: int = 10) -> str:
    """热门搜索排行：返回 Redis 中搜索频次最高的查询。

    Args:
        top_n: 返回条数，默认10
    """
    from core.cache import get_search_cache
    cache = get_search_cache()
    return json.dumps({"hot": cache.hot_queries(top_n)}, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run()
