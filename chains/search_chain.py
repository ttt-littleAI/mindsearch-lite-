"""Phase 1: 基础搜索链
用户输入 → LLM改写关键词 → DuckDuckGo搜索 → LLM生成回答
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from tools.search import web_search


def get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


# 第一步: 把用户问题改写成搜索关键词
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个搜索查询优化专家。把用户的问题改写成适合搜索引擎的关键词，只输出关键词，不要解释。"),
    ("human", "{question}"),
])

# 第二步: 根据搜索结果生成回答
answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个智能搜索助手。根据搜索结果回答用户的问题。\n"
     "要求：\n"
     "1. 基于搜索结果回答，不要编造信息\n"
     "2. 如果搜索结果不足以回答，明确说明\n"
     "3. 引用关键信息的来源\n"
     "4. 用中文回答"),
    ("human",
     "用户问题: {question}\n\n"
     "搜索关键词: {keywords}\n\n"
     "搜索结果:\n{search_results}\n\n"
     "请根据以上搜索结果回答用户的问题。"),
])


def run_search_chain(question: str) -> dict:
    """执行基础搜索链"""
    llm = get_llm()
    parser = StrOutputParser()

    # Step 1: 改写关键词
    keywords = (rewrite_prompt | llm | parser).invoke({"question": question})
    print(f"🔍 搜索关键词: {keywords}")

    # Step 2: 执行搜索
    search_results = web_search.invoke(keywords)
    print(f"📄 搜索结果获取完成")

    # Step 3: 生成回答
    answer = (answer_prompt | llm | parser).invoke({
        "question": question,
        "keywords": keywords,
        "search_results": search_results,
    })

    return {
        "question": question,
        "keywords": keywords,
        "search_results": search_results,
        "answer": answer,
    }


if __name__ == "__main__":
    result = run_search_chain("2024年诺贝尔物理学奖颁给了谁？")
    print(f"\n{'='*60}")
    print(f"问题: {result['question']}")
    print(f"关键词: {result['keywords']}")
    print(f"{'='*60}")
    print(f"回答:\n{result['answer']}")
