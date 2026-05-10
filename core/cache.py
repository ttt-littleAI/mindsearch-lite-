"""Redis 搜索缓存 — 热点查询加速 + 按时效性动态过期

TTL 策略:
  REALTIME  → 10 分钟（新闻/热点变化快）
  STABLE    → 24 小时（知识类相对稳定）
  PERSONAL  → 1 小时（用户文档偶尔更新）

数据结构:
  search:result:{hash}  → JSON，缓存搜索结果
  search:hot            → Sorted Set，按搜索频次排序
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict

import redis

from config.settings import REDIS_HOST, REDIS_PORT, REDIS_DB, REDIS_PASSWORD

logger = logging.getLogger(__name__)

ROUTE_TTL = {
    "REALTIME": 60 * 60,       # 1 小时
    "STABLE": 60 * 60 * 24,    # 24 小时
    "PERSONAL": 60 * 60,       # 1 小时
}
DEFAULT_TTL = 60 * 60  # 1 小时

PREFIX_RESULT = "search:result:"
PREFIX_HOT = "search:hot"


def _query_key(question: str, user_id: str = "default") -> str:
    raw = f"{user_id}:{question.strip().lower()}"
    return hashlib.md5(raw.encode()).hexdigest()


class SearchCache:
    """Redis 搜索缓存管理器"""

    def __init__(self):
        password = REDIS_PASSWORD if REDIS_PASSWORD else None
        self.r = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            password=password,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        self._available = None

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                self.r.ping()
                self._available = True
            except Exception:
                self._available = False
                logger.warning("Redis 不可用，搜索缓存已禁用")
        return self._available

    def get(self, question: str, user_id: str = "default") -> dict | None:
        """查缓存，命中则返回 dict，未命中返回 None"""
        if not self.available:
            return None
        try:
            key = PREFIX_RESULT + _query_key(question, user_id)
            data = self.r.get(key)
            if data:
                self.r.zincrby(PREFIX_HOT, 1, question.strip()[:200])
                logger.info("Redis cache hit: %s", question[:50])
                return json.loads(data)
        except Exception as e:
            logger.warning("Redis 缓存读取失败: %s", e)
        return None

    def put(
        self,
        question: str,
        result: dict,
        route: str = "",
        user_id: str = "default",
    ):
        """存入缓存，TTL 根据路由类型自动设定"""
        if not self.available:
            return
        try:
            key = PREFIX_RESULT + _query_key(question, user_id)
            ttl = ROUTE_TTL.get(route, DEFAULT_TTL)
            self.r.setex(key, ttl, json.dumps(result, ensure_ascii=False))
            self.r.zincrby(PREFIX_HOT, 1, question.strip()[:200])
        except Exception as e:
            logger.warning("Redis 缓存写入失败: %s", e)

    def invalidate(self, question: str, user_id: str = "default"):
        """主动失效某条缓存"""
        if not self.available:
            return
        try:
            key = PREFIX_RESULT + _query_key(question, user_id)
            self.r.delete(key)
        except Exception as e:
            logger.warning("Redis 缓存失效失败: %s", e)

    def hot_queries(self, top_n: int = 20) -> list[dict]:
        """返回搜索频次最高的查询"""
        if not self.available:
            return []
        try:
            items = self.r.zrevrange(PREFIX_HOT, 0, top_n - 1, withscores=True)
            return [{"query": q, "count": int(s)} for q, s in items]
        except Exception:
            return []

    def stats(self) -> dict:
        """缓存统计"""
        if not self.available:
            return {"available": False}
        try:
            info = self.r.info("memory")
            key_count = self.r.dbsize()
            return {
                "available": True,
                "keys": key_count,
                "memory_used": info.get("used_memory_human", "unknown"),
                "hot_queries": self.hot_queries(10),
            }
        except Exception:
            return {"available": False}


_cache_instance: SearchCache | None = None


def get_search_cache() -> SearchCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = SearchCache()
    return _cache_instance
