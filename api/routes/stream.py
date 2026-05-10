"""流式搜索接口 — SSE (Server-Sent Events)

前端通过 EventSource 连接，实时接收搜索过程的每个阶段:
  routing    → 路由决策
  skill      → 技能匹配
  searching  → 子问题搜索中
  ranking    → 精排中
  generating → 生成回答中
  done       → 搜索完成
  error      → 出错
"""

import json
import time
import asyncio
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/search", tags=["搜索"])


@router.get("/stream")
async def stream_search(question: str, user_id: str = "default"):
    """流式搜索：SSE 推送每个阶段的进度和结果"""
    return StreamingResponse(
        _search_stream(question, user_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


async def _search_stream(question: str, user_id: str):
    from core.cache import get_search_cache
    from core.skills import match_skill

    t0 = time.time()

    # Step 1: 技能匹配
    skill = match_skill(question)
    if skill:
        yield _sse_event("skill", {
            "name": skill.display_name,
            "strategy": skill.strategy,
        })

    # Step 2: Redis 缓存检查
    cache = get_search_cache()
    cached = cache.get(question, user_id)
    if cached:
        yield _sse_event("cache_hit", {"source": "redis"})
        yield _sse_event("done", {
            "answer": cached["answer"],
            "metadata": {**cached.get("metadata", {}), "cache": "hit", "duration_ms": 0},
        })
        return

    # Step 3: 路由决策
    from core.router import classify_query, TimelinessCategory

    _STRATEGY_TO_CAT = {
        "web_only": TimelinessCategory.REALTIME,
        "hybrid": TimelinessCategory.STABLE,
        "local_only": TimelinessCategory.PERSONAL,
    }

    # Skill 命中即视为高置信度规则路由，跳过 LLM 分类
    if skill and skill.strategy in _STRATEGY_TO_CAT:
        category = _STRATEGY_TO_CAT[skill.strategy]
        yield _sse_event("routing", {
            "category": category.value,
            "confidence": 1.0,
            "reasoning": f"规则命中: skill={skill.name}",
        })
    else:
        yield _sse_event("routing", {"status": "classifying"})
        route = await asyncio.to_thread(classify_query, question)
        category = TimelinessCategory(route.category)
        yield _sse_event("routing", {
            "category": category.value,
            "confidence": route.confidence,
            "reasoning": route.reasoning,
        })

    # Step 4: 执行搜索
    yield _sse_event("searching", {"status": "started", "strategy": category.value})

    from core.search_engine import search, SearchRequest, SearchMode
    mode_map = {
        TimelinessCategory.REALTIME: SearchMode.WEB,
        TimelinessCategory.STABLE: SearchMode.HYBRID,
        TimelinessCategory.PERSONAL: SearchMode.KNOWLEDGE,
    }

    try:
        result = await asyncio.to_thread(
            search,
            SearchRequest(
                question=question,
                search_mode=mode_map[category],
                user_id=user_id,
            ),
        )
    except Exception as e:
        yield _sse_event("error", {"message": str(e)})
        return

    duration_ms = round((time.time() - t0) * 1000)

    # Step 5: 完成
    yield _sse_event("done", {
        "answer": result.answer,
        "metadata": {**result.metadata, "duration_ms": duration_ms},
    })
