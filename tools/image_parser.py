"""图片理解 — 用 LLM Vision 描述图片内容"""

import base64
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME


def describe_image(file_path: str) -> str:
    """将图片转为 base64 发给 LLM Vision，返回内容描述"""
    path = Path(file_path)
    suffix = path.suffix.lower()
    mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".bmp": "image/bmp", ".webp": "image/webp"}
    mime_type = mime_map.get(suffix, "image/png")

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    msg = HumanMessage(content=[
        {"type": "text", "text": (
            "请详细描述这张图片的内容。"
            "如果是图表，请提取其中的关键数据和趋势。"
            "如果包含文字，请完整提取。"
            "用中文回答，300字以内。"
        )},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
    ])

    response = llm.invoke([msg])
    return response.content
