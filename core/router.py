"""时效性路由器 — 根据查询意图选择搜索策略

分类:
  REALTIME  — 需要最新信息（新闻、实时数据、最近事件）→ 直接网搜
  STABLE    — 稳定知识（概念、原理、对比分析）→ 先查本地库，不够再补搜
  PERSONAL  — 用户私有文档相关 → 只查用户上传的文档
"""

import re
from dataclasses import dataclass
from enum import Enum

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME


class TimelinessCategory(str, Enum):
    REALTIME = "REALTIME"
    STABLE = "STABLE"
    PERSONAL = "PERSONAL"


@dataclass
class RouteDecision:
    category: str
    reasoning: str
    confidence: float


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个查询意图分类器。根据用户的问题，判断它属于以下哪个类别。\n\n"
     "**REALTIME** — 需要最新、实时的信息才能回答\n"
     "  例: 今天的天气、最新的股价、刚发布的新闻、当前的政策\n"
     "  信号词: 今天、最新、刚刚、目前、现在、最近（一周内）\n\n"
     "**STABLE** — 稳定的知识，不会频繁变化\n"
     "  例: 什么是机器学习、Python和Java的区别、光合作用的原理\n"
     "  信号词: 什么是、如何、为什么、对比、原理、概念\n\n"
     "**PERSONAL** — 涉及用户自己的文档或私有数据\n"
     "  例: 我上传的论文里提到了什么、我的报告中的数据、我的文档\n"
     "  信号词: 我的、我上传的、我的文档、我的文件\n\n"
     "严格按以下格式输出（三行，不要多余内容）:\n"
     "CATEGORY: REALTIME 或 STABLE 或 PERSONAL\n"
     "REASON: 一句话理由\n"
     "CONFIDENCE: 0到1之间的数字"),
    ("human", "问题: {question}"),
])

_CATEGORY_PATTERN = re.compile(r"CATEGORY:\s*(REALTIME|STABLE|PERSONAL)", re.IGNORECASE)
_REASON_PATTERN = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)
_CONFIDENCE_PATTERN = re.compile(r"CONFIDENCE:\s*([\d.]+)", re.IGNORECASE)

# 规则前置短路：高置信度信号词命中即返回，跳过 LLM 调用
_PERSONAL_RE = re.compile(r"我的(文档|文件|论文|报告|笔记|资料)|文档里|文件中|报告中|上传的")
_REALTIME_RE = re.compile(r"今天|昨天|刚刚|刚才|最新|目前|实时|现在|本周|这周|本月|最近")


def classify_query(question: str) -> RouteDecision:
    # Layer 0: 规则前置 — 强信号词直接路由，不调 LLM
    if _PERSONAL_RE.search(question):
        return RouteDecision("PERSONAL", "规则命中:私有文档信号词", 1.0)
    if _REALTIME_RE.search(question):
        return RouteDecision("REALTIME", "规则命中:时效性信号词", 0.95)

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    try:
        raw = (ROUTER_PROMPT | llm | StrOutputParser()).invoke({"question": question})
        return _parse_decision(raw)
    except Exception:
        return RouteDecision(
            category="REALTIME",
            reasoning="分类失败，默认使用实时搜索",
            confidence=0.5,
        )


def _parse_decision(text: str) -> RouteDecision:
    cat_match = _CATEGORY_PATTERN.search(text)
    reason_match = _REASON_PATTERN.search(text)
    conf_match = _CONFIDENCE_PATTERN.search(text)

    category = cat_match.group(1).upper() if cat_match else "REALTIME"
    reasoning = reason_match.group(1).strip() if reason_match else "无法解析理由"
    try:
        confidence = float(conf_match.group(1)) if conf_match else 0.5
    except ValueError:
        confidence = 0.5

    if category not in ("REALTIME", "STABLE", "PERSONAL"):
        category = "REALTIME"

    return RouteDecision(category=category, reasoning=reasoning, confidence=confidence)
