"""Phase 2: ReAct Agent
用 LangGraph 构建 ReAct Agent，LLM 自主决定是否搜索、搜几次、是否追问。
"""

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from tools.search import web_search


SYSTEM_PROMPT = """你是一个智能搜索助手 MindSearch-Lite。

你的工作方式：
1. 分析用户的问题，判断是否需要搜索
2. 如果需要搜索，使用 web_search 工具搜索相关信息
3. 你可以多次搜索，从不同角度获取信息
4. 如果第一次搜索结果不够，可以换关键词再搜
5. 综合所有搜索结果，给出准确、有条理的回答

回答要求：
- 基于搜索结果，不要编造信息
- 引用关键信息来源
- 用中文回答
- 结构化输出，使用标题和列表
"""


def create_search_agent():
    """创建 ReAct 搜索 Agent"""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    agent = create_react_agent(
        model=llm,
        tools=[web_search],
        prompt=SYSTEM_PROMPT,
    )
    return agent


def run_react_agent(question: str) -> str:
    """运行 ReAct Agent"""
    agent = create_search_agent()

    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]}
    )

    # 提取最终回答
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    answer = run_react_agent("对比一下 2024 年 OpenAI 和 Anthropic 发布的主要模型")
    print(answer)
