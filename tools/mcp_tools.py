"""MCP Client 工具集 — Agent 可调用的外部工具

工具注册表：每个工具定义 JSON Schema + 执行函数，
LLM 通过 function calling 决定调用哪个工具。

内置工具:
  url_scrape   — 网页正文抓取（搜索结果摘要不够时抓全文）
  calculator   — 数学计算（搜到数据需要换算/计算）
  code_runner  — Python 代码执行（技术问题代码验证）
  translator   — 中英翻译（跨语言搜索增强）
  datetime_now — 日期时间（时效性路由需要的时间上下文）
  redis_query  — Redis 缓存操作（读写缓存、热搜、并发限流）
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Callable

import requests


@dataclass
class MCPTool:
    name: str
    description: str
    parameters: dict
    func: Callable


TOOL_REGISTRY: dict[str, MCPTool] = {}


def register(name: str, description: str, parameters: dict):
    """装饰器：注册工具"""
    def wrapper(func: Callable):
        TOOL_REGISTRY[name] = MCPTool(
            name=name,
            description=description,
            parameters=parameters,
            func=func,
        )
        return func
    return wrapper


def call_tool(name: str, arguments: dict) -> str:
    tool = TOOL_REGISTRY.get(name)
    if not tool:
        return json.dumps({"error": f"工具 {name} 不存在"}, ensure_ascii=False)
    try:
        result = tool.func(**arguments)
        return json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, dict) else str(result)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def list_tools_schema() -> list[dict]:
    """返回所有工具的 JSON Schema（供 LLM function calling 使用）"""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in TOOL_REGISTRY.values()
    ]


# ── 工具实现 ──


@register(
    name="url_scrape",
    description="抓取网页正文内容。搜索结果只有摘要时，用这个工具获取完整文章内容",
    parameters={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "要抓取的网页 URL"},
            "max_chars": {"type": "integer", "description": "最大返回字符数", "default": 3000},
        },
        "required": ["url"],
    },
)
def url_scrape(url: str, max_chars: int = 3000) -> dict:
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (compatible; MindSearch/2.0)",
        })
        resp.raise_for_status()
        resp.encoding = resp.apparent_encoding

        text = resp.text
        text = re.sub(r'<script[^>]*>[\s\S]*?</script>', '', text)
        text = re.sub(r'<style[^>]*>[\s\S]*?</style>', '', text)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return {"url": url, "content": text[:max_chars], "length": len(text)}
    except Exception as e:
        return {"url": url, "error": str(e)}


@register(
    name="calculator",
    description="数学计算器。搜索到的数据需要计算、换算、统计时使用",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "数学表达式，例如 '(1024 * 768) / 1e6'"},
        },
        "required": ["expression"],
    },
)
def calculator(expression: str) -> dict:
    safe_chars = set("0123456789.+-*/%() ,eE")
    safe_funcs = {"abs", "round", "min", "max", "sum", "pow",
                  "sqrt", "log", "log2", "log10", "sin", "cos", "tan", "pi"}

    tokens = re.findall(r'[a-zA-Z_]+', expression)
    for token in tokens:
        if token not in safe_funcs:
            return {"error": f"不允许的函数: {token}"}

    namespace = {name: getattr(math, name) for name in dir(math) if not name.startswith('_')}
    namespace.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})

    try:
        result = eval(expression, {"__builtins__": {}}, namespace)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


@register(
    name="code_runner",
    description="执行 Python 代码片段。技术问题需要代码验证时使用，超时 10 秒",
    parameters={
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python 代码"},
        },
        "required": ["code"],
    },
)
def code_runner(code: str) -> dict:
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=10,
        )
        return {
            "stdout": result.stdout[:2000],
            "stderr": result.stderr[:500],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"error": "执行超时（10秒）"}
    except Exception as e:
        return {"error": str(e)}


@register(
    name="translator",
    description="中英互译。跨语言搜索时，将英文论文/网页翻译成中文，或将中文查询翻译成英文搜索",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "要翻译的文本"},
            "target_lang": {"type": "string", "description": "目标语言: zh（中文）或 en（英文）", "default": "zh"},
        },
        "required": ["text"],
    },
)
def translator(text: str, target_lang: str = "zh") -> dict:
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

    lang_name = "中文" if target_lang == "zh" else "英文"

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        temperature=0,
    )
    result = (llm | StrOutputParser()).invoke(
        f"将以下内容翻译成{lang_name}，只输出译文，不要解释:\n\n{text}"
    )
    return {"original": text[:200], "translated": result, "target": target_lang}


@register(
    name="datetime_now",
    description="获取当前日期时间。时效性路由判断'最近''今天'等相对时间时需要",
    parameters={
        "type": "object",
        "properties": {},
    },
)
def datetime_now() -> dict:
    now = datetime.now()
    return {
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()],
        "timestamp": int(now.timestamp()),
    }


@register(
    name="redis_query",
    description="Redis 缓存操作：查询缓存、热搜排行、限流检查。用于搜索加速和并发控制",
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "操作类型: get_cache(查缓存) / hot(热搜) / rate_check(限流检查) / stats(统计)",
                "enum": ["get_cache", "hot", "rate_check", "stats"],
            },
            "query": {"type": "string", "description": "查询内容（get_cache 时使用）", "default": ""},
            "user_id": {"type": "string", "description": "用户ID（rate_check 时使用）", "default": "default"},
        },
        "required": ["action"],
    },
)
def redis_query(action: str, query: str = "", user_id: str = "default") -> dict:
    from core.cache import get_search_cache
    cache = get_search_cache()
    if not cache.available:
        return {"error": "Redis 不可用"}

    if action == "get_cache":
        result = cache.get(query, user_id)
        return {"hit": result is not None, "data": result} if query else {"error": "缺少 query 参数"}

    elif action == "hot":
        return {"hot_queries": cache.hot_queries(20)}

    elif action == "rate_check":
        try:
            key = f"rate_limit:{user_id}"
            count = cache.r.get(key)
            if count is None:
                cache.r.setex(key, 60, 1)
                return {"user_id": user_id, "requests_in_minute": 1, "limit": 30, "allowed": True}
            count = int(count)
            if count >= 30:
                return {"user_id": user_id, "requests_in_minute": count, "limit": 30, "allowed": False}
            cache.r.incr(key)
            return {"user_id": user_id, "requests_in_minute": count + 1, "limit": 30, "allowed": True}
        except Exception as e:
            return {"error": str(e)}

    elif action == "stats":
        return cache.stats()

    return {"error": f"未知操作: {action}"}
