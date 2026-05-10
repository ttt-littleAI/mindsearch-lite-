"""MindSearch v2 — Gradio 入口"""

import gradio as gr
from core.search_engine import search as core_search, SearchRequest


def search(question: str, history: list) -> str:
    if not question.strip():
        return "请输入一个问题。"

    try:
        result = core_search(SearchRequest(question=question))
        return result.answer
    except Exception as e:
        return f"搜索出错: {str(e)}"


def create_ui():
    """创建 Gradio 界面"""
    with gr.Blocks(
        title="MindSearch-Lite",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🔍 MindSearch-Lite\n"
            "基于 LangGraph 的多 Agent 智能搜索引擎\n\n"
            "**工作流程:** 问题拆解 → 并行搜索 → RAG增强 → 综合回答"
        )

        chatbot = gr.ChatInterface(
            fn=search,
            type="messages",
            examples=[
                "对比 2024 年 OpenAI 和 Anthropic 发布的主要模型",
                "中国和美国在人工智能领域的最新政策有什么区别？",
                "LangGraph 和 CrewAI 哪个更适合做多 Agent 系统？",
                "2024年诺贝尔物理学奖颁给了谁？为什么？",
            ],
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)
