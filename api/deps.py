"""依赖注入 — FastAPI Depends 使用"""

from agents.rag import get_memory, SearchMemory


def get_search_memory() -> SearchMemory:
    return get_memory()
