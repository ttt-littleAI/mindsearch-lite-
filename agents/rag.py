"""Phase 4: RAG + Memory — 向量知识库 + 对话记忆

基于 core.vector_store 的多集合存储:
- search_cache: Searcher 搜索结果缓存
- document_chunks: 用户文档检索
- 对话记忆跨轮次保持
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from core.vector_store import get_vector_store


class SearchMemory:
    """搜索记忆 — 多集合向量库 + 对话历史"""

    def __init__(self):
        self.store = get_vector_store()
        self.chat_history: list = []

    def add_documents(self, docs: list[Document]):
        self.store.add_documents(docs)

    def search(self, query: str, k: int = 3) -> list[Document]:
        return self.store.search(query, k=k)

    def add_to_chat_history(self, role: str, content: str):
        if role == "human":
            self.chat_history.append(HumanMessage(content=content))
        else:
            self.chat_history.append(AIMessage(content=content))

    def get_chat_history(self, max_turns: int = 10):
        return self.chat_history[-(max_turns * 2):]


_memory_instance = None


def get_memory() -> SearchMemory:
    """获取全局 SearchMemory 单例"""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = SearchMemory()
    return _memory_instance
