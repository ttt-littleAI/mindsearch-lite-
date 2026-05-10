"""缓存管理接口 — 热搜/统计/失效"""

from fastapi import APIRouter

from core.cache import get_search_cache

router = APIRouter(prefix="/cache", tags=["cache"])


@router.get("/stats")
def cache_stats():
    """缓存统计 + 热搜 top10"""
    return get_search_cache().stats()


@router.get("/hot")
def hot_queries(top_n: int = 20):
    """热门搜索排行"""
    return get_search_cache().hot_queries(top_n)


@router.delete("/invalidate")
def invalidate(question: str, user_id: str = "default"):
    """手动失效某条缓存"""
    get_search_cache().invalidate(question, user_id)
    return {"status": "ok"}
