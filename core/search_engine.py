"""统一搜索入口 — 所有入口（FastAPI / Gradio / CLI / MCP）都走这里

三层管线:
  Layer 1: Input Understanding  — 意图识别 + 时效性分类
  Layer 2: Routing              — 根据时效性选择搜索策略
  Layer 3: Search + Fusion      — 执行搜索 + 融合结果
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from core.exceptions import SearchError
from core.router import classify_query, TimelinessCategory

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    AUTO = "auto"
    WEB = "web"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"


@dataclass
class SearchRequest:
    question: str
    files: list[str] = field(default_factory=list)
    search_mode: SearchMode = SearchMode.AUTO
    user_id: str = "default"


@dataclass
class SearchResponse:
    answer: str
    citations: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


_STRATEGY_TO_CATEGORY = {
    "web_only": TimelinessCategory.REALTIME,
    "hybrid": TimelinessCategory.STABLE,
    "local_only": TimelinessCategory.PERSONAL,
}


def search(request: SearchRequest) -> SearchResponse:
    """同步搜索入口"""

    # --- Layer 1: Input Understanding ---
    from core.skills import match_skill
    matched_skill = match_skill(request.question)
    if matched_skill:
        print(f"\n🎯 技能匹配: {matched_skill.display_name} (策略: {matched_skill.strategy})")

    if request.search_mode == SearchMode.AUTO:
        # Skill 命中即视为高置信度规则路由，跳过 LLM 分类
        if matched_skill and matched_skill.strategy in _STRATEGY_TO_CATEGORY:
            category = _STRATEGY_TO_CATEGORY[matched_skill.strategy]
            print(f"🧭 路由决策: {category.value} (来源: skill={matched_skill.name}, 短路 LLM)")
        else:
            route = classify_query(request.question)
            category = TimelinessCategory(route.category)
            print(f"🧭 路由决策: {category.value} (置信度: {route.confidence:.0%})")
            print(f"   理由: {route.reasoning}")

            # 置信度不足时引导用户澄清（仅 LLM 路径会触发）
            if route.confidence < 0.6:
                clarify_msg = _build_clarify_message(request.question, route)
                return SearchResponse(
                    answer=clarify_msg,
                    metadata={
                        "route": "CLARIFY",
                        "confidence": route.confidence,
                        "reasoning": route.reasoning,
                        "duration_ms": 0,
                    },
                )
    else:
        category = {
            SearchMode.WEB: TimelinessCategory.REALTIME,
            SearchMode.KNOWLEDGE: TimelinessCategory.PERSONAL,
            SearchMode.HYBRID: TimelinessCategory.STABLE,
        }[request.search_mode]
        print(f"\n🧭 手动指定路由: {category.value}")

    route_name = category.value

    # --- Redis 缓存：先查 ---
    from core.cache import get_search_cache
    cache = get_search_cache()
    cached = cache.get(request.question, request.user_id)
    if cached:
        print(f"   ⚡ Redis 缓存命中")
        return SearchResponse(
            answer=cached["answer"],
            citations=cached.get("citations", []),
            metadata={
                **cached.get("metadata", {}),
                "cache": "hit",
                "duration_ms": 0,
            },
        )

    # --- Layer 2+3: Routing + Search ---
    t0 = time.time()
    if category == TimelinessCategory.REALTIME:
        resp = _route_realtime(request, matched_skill)
    elif category == TimelinessCategory.STABLE:
        resp = _route_stable(request, matched_skill)
    else:
        resp = _route_personal(request, matched_skill)
    duration_ms = (time.time() - t0) * 1000

    # --- Redis 缓存：存入 ---
    cache.put(
        question=request.question,
        result={
            "answer": resp.answer,
            "citations": resp.citations,
            "metadata": resp.metadata,
        },
        route=route_name,
        user_id=request.user_id,
    )

    # RAGAS 四维评估（同步调用 LLM 评分，失败静默；只对有 citations 的搜索评估）
    eval_scores = None
    if resp.citations:
        try:
            from core.evaluator import evaluate
            contexts = [c.get("snippet", "") for c in resp.citations if c.get("snippet")]
            eval_result = evaluate(request.question, resp.answer, contexts)
            eval_scores = {
                "faithfulness": eval_result.faithfulness,
                "answer_relevancy": eval_result.answer_relevancy,
                "context_precision": eval_result.context_precision,
                "context_recall": eval_result.context_recall,
                "overall": eval_result.overall,
            }
            print(f"   📊 RAGAS: overall={eval_result.overall:.2f} (F={eval_result.faithfulness:.2f}/R={eval_result.answer_relevancy:.2f})")
        except Exception as e:
            print(f"   ⚠️ RAGAS 评估失败（已跳过）: {e}")

    # 记录搜索日志（MySQL 不可用时静默跳过）
    try:
        from core.database import log_search
        log_search(
            user_id=request.user_id,
            question=request.question,
            route=resp.metadata.get("route", ""),
            strategy=resp.metadata.get("strategy", ""),
            source=resp.metadata.get("source", ""),
            answer_length=len(resp.answer),
            duration_ms=duration_ms,
            eval_scores=eval_scores,
        )
    except Exception as e:
        logger.warning("MySQL log_search 写入失败: %s", e)

    if eval_scores:
        resp.metadata["eval"] = eval_scores

    resp.metadata["duration_ms"] = round(duration_ms)
    resp.metadata["cache"] = "miss"
    return resp


def _route_realtime(request: SearchRequest, skill=None) -> SearchResponse:
    """REALTIME: 纯网络搜索 Agent 流"""
    from agents.multi_agent import run_mindsearch

    try:
        answer, citations = run_mindsearch(
            request.question,
            strategy="web_only",
            user_id=request.user_id,
            skill=skill,
        )
    except Exception as e:
        raise SearchError(f"搜索失败: {e}") from e

    return SearchResponse(
        answer=answer,
        citations=citations,
        metadata={
            "route": "REALTIME",
            "strategy": "web_only",
            "user_id": request.user_id,
            "skill": skill.name if skill else None,
        },
    )


def _build_clarify_message(question: str, route) -> str:
    """置信度不足时生成澄清引导"""
    return (
        f"我不太确定你想搜索的方向，你的问题是：「{question}」\n\n"
        f"请问你想：\n"
        f"1. 🔍 **搜索最新信息**（新闻、热点、实时数据）\n"
        f"2. 📚 **查询知识原理**（概念解释、技术对比、学术内容）\n"
        f"3. 📄 **查找你的文档**（在你上传的文件中搜索）\n\n"
        f"请回复数字 1/2/3，或补充更多描述帮助我理解你的需求。"
    )


def _route_stable(request: SearchRequest, skill=None) -> SearchResponse:
    """STABLE: 本地知识库 + 网络并行搜索，合并排序"""
    from agents.multi_agent import run_mindsearch

    try:
        answer, citations = run_mindsearch(
            request.question,
            strategy="hybrid",
            user_id=request.user_id,
            skill=skill,
        )
    except Exception as e:
        raise SearchError(f"搜索失败: {e}") from e

    return SearchResponse(
        answer=answer,
        citations=citations,
        metadata={
            "route": "STABLE",
            "strategy": "hybrid",
            "user_id": request.user_id,
            "skill": skill.name if skill else None,
        },
    )


def _route_personal(request: SearchRequest, skill=None) -> SearchResponse:
    """PERSONAL: 只查用户上传的文档"""
    from core.vector_store import get_vector_store

    store = get_vector_store()
    docs = store.search_documents(request.question, k=5, user_id=request.user_id)

    if not docs:
        return SearchResponse(
            answer="未在你的文档中找到相关内容。请先上传文档，或换用其他搜索模式。",
            metadata={
                "route": "PERSONAL",
                "source": "no_documents",
                "user_id": request.user_id,
                "skill": skill.name if skill else None,
            },
        )

    context = "\n\n".join(doc.page_content for doc in docs)
    answer = _synthesize_from_context(request.question, context, docs, skill)

    # 把本地命中的 docs 转成与 REALTIME/STABLE 一致的全局引用结构
    citations = [
        {
            "global_index": i + 1,
            "title": doc.metadata.get("source", "本地文档"),
            "url": "",  # 本地文档无 URL
            "snippet": doc.page_content[:300],
            "sub_question": request.question,
        }
        for i, doc in enumerate(docs)
    ]

    return SearchResponse(
        answer=answer,
        citations=citations,
        metadata={
            "route": "PERSONAL",
            "source": "user_documents",
            "doc_count": len(docs),
            "user_id": request.user_id,
            "skill": skill.name if skill else None,
        },
    )


def _synthesize_from_context(question: str, context: str, docs: list, skill=None) -> str:
    """基于本地知识生成回答"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    sources = "\n".join(
        f"[{i+1}] {doc.metadata.get('source', '本地文档')}"
        for i, doc in enumerate(docs)
    )

    # 命中 skill 时用 skill.system_prompt 覆盖默认 prompt（如 document_qa 的严格基于文档要求）
    system_prompt = skill.system_prompt if skill else (
        "你是 MindSearch 搜索助手。基于提供的文档内容回答问题。\n"
        "- 只使用提供的内容，不要编造\n"
        "- 关键事实后标注来源 [n]\n"
        "- 300字以内，简洁准确\n"
        "- 用中文回答"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "问题: {question}\n\n"
         "文档内容:\n{context}\n\n"
         "来源:\n{sources}\n\n"
         "请回答:"),
    ])

    return (prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "context": context,
        "sources": sources,
    })


async def search_async(request: SearchRequest) -> SearchResponse:
    """异步搜索入口 — FastAPI 使用"""
    return await asyncio.to_thread(search, request)
