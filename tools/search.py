"""搜索工具 — 封装 DuckDuckGo 搜索，返回结构化结果"""

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool


ddg_search = DuckDuckGoSearchResults(max_results=5)
ddg_wrapper = DuckDuckGoSearchAPIWrapper(max_results=8)


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息。输入应该是一个搜索关键词。"""
    results = ddg_search.invoke(query)
    return results


def web_search_structured(query: str) -> list[dict]:
    """搜索互联网，返回结构化结果列表。

    每条结果包含:
      - title: 页面标题
      - url: 链接
      - snippet: 摘要片段
    """
    try:
        results = ddg_wrapper.results(query, max_results=8)
        return [
            {
                "title": r.get("title", ""),
                "url": r.get("link", ""),
                "snippet": r.get("snippet", r.get("body", "")),
            }
            for r in results
        ]
    except Exception:
        # 降级到旧接口
        raw = ddg_search.invoke(query)
        return [{"title": "", "url": "", "snippet": raw}]
