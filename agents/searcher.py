"""Searcher Agent — 多策略搜索 + 局部引用系统

策略:
  web_only — 纯网络搜索（REALTIME 路由）
  hybrid   — 本地 + 网络并行，合并结果（STABLE 路由）

Stage 1 (Search): 根据策略获取候选结果
Stage 2 (Select): LLM 筛选最相关的结果
Stage 3 (Summarize): 基于筛选结果生成带局部引用的总结
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from tools.search import web_search_structured


# ── Stage 2: Select — 筛选最相关结果 ──────────────────────────

class SelectedIndices(BaseModel):
    """筛选结果"""
    selected: list[int] = Field(description="选中的结果编号列表（从0开始）")
    reasoning: str = Field(description="筛选理由")


SELECT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个搜索结果筛选专家。给定一个子问题和一组搜索结果，"
     "从中选出与子问题最相关、信息量最大的结果。\n\n"
     "规则:\n"
     "1. 只选与子问题直接相关的结果\n"
     "2. 优先选信息量大、来源可靠的结果\n"
     "3. 通常选 3-5 条，去掉明显无关或重复的\n"
     "4. 返回选中结果的编号（从0开始）"),
    ("human",
     "子问题: {sub_question}\n\n"
     "候选搜索结果:\n{candidates}\n\n"
     "请选出最相关的结果。"),
])


# ── Stage 3: Summarize — 带局部引用的总结 ─────────────────────

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个搜索结果分析专家。根据搜索结果，针对子问题给出简洁准确的总结。\n\n"
     "**引用规则（必须严格遵守）:**\n"
     "1. 每个具体事实后面紧跟 [n] 引用，n 是来源编号\n"
     "2. 引用要精确绑定到具体事实，不要在句末笼统堆砌\n"
     "3. 不要编造来源中没有的信息或数字\n"
     "4. 数字只引用来源中明确出现的，不确定的用\"约\"\"据报道\"限定\n"
     "5. 总结控制在 100-200 字以内，抓重点\n\n"
     "正确: \"OpenAI 发布 GPT-4 [1]，Google 推出 Gemini [3]\"\n"
     "错误: \"各大公司纷纷发布新模型 [1][2][3]\""),
    ("human",
     "子问题: {sub_question}\n\n"
     "来源列表:\n{sources}\n\n"
     "请精炼总结（100-200字），每个事实紧跟引用编号 [n]。"),
])


def _fetch_local_results(sub_question: str, user_id: str | None = None) -> list[dict]:
    """从本地向量库检索"""
    from core.vector_store import get_vector_store
    store = get_vector_store()
    docs = store.search(sub_question, k=5)
    if user_id:
        user_docs = store.search_documents(sub_question, k=3, user_id=user_id)
        docs.extend(user_docs)
    return [
        {
            "title": doc.metadata.get("source", "本地知识库"),
            "url": f"local://{doc.metadata.get('source', 'knowledge')}",
            "snippet": doc.page_content,
        }
        for doc in docs
        if doc.page_content.strip()
    ]


def search_and_summarize(
    sub_question: str,
    strategy: str = "web_only",
    user_id: str | None = None,
) -> dict:
    """多策略搜索一个子问题：search → select → summarize（带局部引用）

    strategy:
        web_only — 纯网络搜索
        hybrid   — 本地+网络合并

    返回:
        {
            "sub_question": str,
            "summary": str,
            "citations": list[dict],
            "raw_results": list[dict],
            "source_type": str,       # web / local / hybrid
        }
    """
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    parser = StrOutputParser()

    # ── Stage 1: Search (策略分发) ───────────────────────────
    source_type = "web"
    raw_results = []

    if strategy == "hybrid":
        local_results = _fetch_local_results(sub_question, user_id)
        web_results = web_search_structured(sub_question)
        raw_results = local_results + web_results
        source_type = "hybrid"
    else:  # web_only
        raw_results = web_search_structured(sub_question)
        source_type = "web"

    if not raw_results:
        return {
            "sub_question": sub_question,
            "summary": "未找到相关搜索结果。",
            "citations": [],
            "raw_results": [],
            "source_type": source_type,
        }

    # ── Stage 2: Select (Rerank 精排) ──────────────────────────
    from core.reranker import rerank

    snippets = [r["snippet"][:300] for r in raw_results]
    reranked = rerank(sub_question, snippets, top_n=min(5, len(raw_results)))
    selected_results = [raw_results[r.index] for r in reranked]

    # ── Stage 3: Summarize with citations ────────────────────
    citations = []
    sources_text_parts = []
    for local_idx, r in enumerate(selected_results, 1):
        citations.append({
            "index": local_idx,
            "title": r["title"],
            "url": r["url"],
            "snippet": r["snippet"][:300],
        })
        sources_text_parts.append(
            f"[{local_idx}] {r['title']}\n"
            f"    链接: {r['url']}\n"
            f"    内容: {r['snippet'][:300]}"
        )

    sources_text = "\n\n".join(sources_text_parts)

    summary = (SUMMARIZE_PROMPT | llm | parser).invoke({
        "sub_question": sub_question,
        "sources": sources_text,
    })

    return {
        "sub_question": sub_question,
        "summary": summary,
        "citations": citations,
        "raw_results": raw_results,
        "source_type": source_type,
    }
