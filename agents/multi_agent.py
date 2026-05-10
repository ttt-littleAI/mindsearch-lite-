"""Phase 3+4+5+6: 多Agent协作 + RAG + Reflection + 引用系统 + Graph Planner

工作流:
1. Planner 拆解问题为 DAG（Graph Schema），按拓扑序分层
2. RAG Retriever 先从本地向量库查已有知识
3. Searcher 按层并行搜索（两阶段: search → select → summarize），每条结果带局部引用
4. Citation Mapper 将所有局部引用统一映射为全局编号
5. Evaluator 判断够不够 → 不够回到 Planner
6. Synthesizer 汇总（使用全局引用编号）
7. Reflector 反思 → 不满意回到 Synthesizer 修正

图结构:
  START → planner → rag_retrieve → searcher → rag_store → citation_map → evaluator
            ↑                                                                 |
            └──────────────── "continue" ─────────────────────────────────────┘
                                                                               |
                                                                        "finish" → synthesizer → reflector
                                                                                      ↑              |
                                                                                      └── "revise" ──┘
                                                                                                      |
                                                                                               "accept" → END
"""

import asyncio
import logging
import operator
import re
from typing import Annotated, Literal
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from config import EVAL_API_KEY, EVAL_BASE_URL, EVAL_MODEL_NAME
from agents.planner import plan_search_graph, replan_search, topological_sort, SearchGraph
from agents.searcher import search_and_summarize
from agents.rag import get_memory
from core.memory import get_ai_memory


MAX_ROUNDS = 3
MAX_REVISIONS = 2


# ── State 定义 ──────────────────────────────────────────────

class MindSearchState(TypedDict):
    question: str
    sub_questions: list[str]
    search_graph: dict | None                                   # Graph Schema (序列化)
    search_results: Annotated[list[dict], operator.add]         # 网络搜索结果(可追加)
    local_knowledge: list[str]
    scratchpad: Annotated[list[str], operator.add]
    current_round: int
    should_continue: bool
    final_answer: str
    critique: str
    revision_count: int
    should_revise: bool
    # ── 引用系统 ──
    global_citations: list[dict]                                # 全局引用列表
    # citation_map_node 写入：summary 中已被替换为全局编号的 search_results
    # 用独立字段而非覆盖 search_results，因为后者用了 operator.add 累加（多轮搜索追加）
    mapped_search_results: list[dict]
    # ── 路由策略 ──
    search_strategy: str                                        # web_only / hybrid
    user_id: str
    # ── Skill 上下文（命中规则时由 search_engine 注入；未命中为 None） ──
    skill_system_prompt: str                                    # 覆盖 Synthesizer 默认 prompt
    skill_name: str                                             # 用于日志和 metadata
    skill_max_sub_questions: int                                # 0=未设置则用 Planner 默认 5
    skill_output_format: str                                    # markdown / json / plain，空字符串表未设置


# ── 引用映射工具 ─────────────────────────────────────────────

def _build_global_citations(search_results: list[dict]) -> tuple[list[dict], list[dict]]:
    """将所有 Searcher 的局部引用统一映射为全局编号。

    返回:
        (global_citations, updated_search_results)

    global_citations: [{global_index, title, url, snippet, sub_question}, ...]
    updated_search_results: 每条结果的 summary 中 [局部n] 被替换为 [全局N]
    """
    global_citations = []
    updated_results = []
    # 用 url 去重
    seen_urls = {}

    for result in search_results:
        local_citations = result.get("citations", [])
        local_to_global = {}

        for lc in local_citations:
            url = lc.get("url", "")
            # 相同 URL 复用全局编号
            if url and url in seen_urls:
                local_to_global[lc["index"]] = seen_urls[url]
            else:
                global_idx = len(global_citations) + 1
                global_citations.append({
                    "global_index": global_idx,
                    "title": lc.get("title", ""),
                    "url": url,
                    "snippet": lc.get("snippet", ""),
                    "sub_question": result["sub_question"],
                })
                local_to_global[lc["index"]] = global_idx
                if url:
                    seen_urls[url] = global_idx

        # 替换 summary 中的局部引用 [n] → [全局N]
        summary = result["summary"]
        if local_to_global:
            def _replace(m):
                local_idx = int(m.group(1))
                global_idx = local_to_global.get(local_idx, local_idx)
                return f"[{global_idx}]"
            summary = re.sub(r"\[(\d+)\]", _replace, summary)

        updated_results.append({
            **result,
            "summary": summary,
        })

    return global_citations, updated_results


# ── Node 函数 ───────────────────────────────────────────────

def planner_node(state: MindSearchState) -> dict:
    """Planner: 首轮用 Graph Schema 拆解，后续轮追加"""
    question = state["question"]
    current_round = state.get("current_round", 0) + 1
    search_results = state.get("search_results", [])

    if not search_results:
        print(f"\n🧠 [轮次 {current_round}] Planner 正在分析问题（Graph Schema）: {question}")
        skill_max = state.get("skill_max_sub_questions", 0)
        if skill_max and skill_max > 0:
            print(f"   📐 Skill 限定子问题上限: {skill_max}")
            graph = plan_search_graph(question, max_questions=skill_max)
        else:
            graph = plan_search_graph(question)
        layers = topological_sort(graph)
        sub_questions = [n.question for n in graph.nodes]

        print(f"📋 拆解为 {len(graph.nodes)} 个子问题，{len(layers)} 层 DAG:")
        for layer_idx, layer in enumerate(layers):
            for n in layer:
                deps = f" ← 依赖 [{', '.join(str(d) for d in n.dependencies)}]" if n.dependencies else ""
                print(f"   层{layer_idx}: [{n.id}] {n.question}{deps}")

        graph_dict = {
            "nodes": [{"id": n.id, "question": n.question, "dependencies": n.dependencies} for n in graph.nodes],
            "reasoning": graph.reasoning,
        }
    else:
        print(f"\n🧠 [轮次 {current_round}] Planner 正在追加搜索...")
        replan = replan_search(question, search_results)
        sub_questions = replan.new_sub_questions
        graph_dict = state.get("search_graph")
        print(f"📋 追加 {len(sub_questions)} 个新子问题:")
        for i, q in enumerate(sub_questions, 1):
            print(f"   {i}. {q}")

    scratchpad_entry = (
        f"[Planner 轮次{current_round}] "
        + ("Graph拆解: " if not search_results else "追加搜索: ")
        + " | ".join(sub_questions)
    )

    return {
        "sub_questions": sub_questions,
        "search_graph": graph_dict,
        "current_round": current_round,
        "scratchpad": [scratchpad_entry],
    }


def rag_retrieve_node(state: MindSearchState) -> dict:
    """RAG Retriever: 搜索前先从本地向量库查已有知识"""
    memory = get_memory()
    question = state["question"]
    sub_questions = state["sub_questions"]
    current_round = state.get("current_round", 1)

    queries = [question] + sub_questions
    all_docs = []
    seen = set()
    for q in queries:
        docs = memory.search(q, k=2)
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                all_docs.append(doc.page_content)

    if all_docs:
        print(f"\n📚 [轮次 {current_round}] 从本地知识库检索到 {len(all_docs)} 条相关记录")
    else:
        print(f"\n📚 [轮次 {current_round}] 本地知识库暂无相关记录")

    scratchpad_entry = (
        f"[RAG Retrieve 轮次{current_round}] "
        f"检索到 {len(all_docs)} 条本地知识"
        + (f"，摘要: {'; '.join(d[:50] for d in all_docs[:3])}" if all_docs else "")
    )

    return {"local_knowledge": all_docs, "scratchpad": [scratchpad_entry]}


def searcher_node(state: MindSearchState) -> dict:
    """Searcher: 按 DAG 分层并行搜索（策略: web_only / hybrid）"""
    sub_questions = state["sub_questions"]
    current_round = state.get("current_round", 1)
    graph_dict = state.get("search_graph")
    strategy = state.get("search_strategy", "web_only")
    user_id = state.get("user_id", "default")

    strategy_label = {"web_only": "网络", "hybrid": "混合"}
    print(f"\n🔍 [轮次 {current_round}] Searcher [{strategy_label.get(strategy, strategy)}策略]")

    if graph_dict and not state.get("search_results"):
        from agents.planner import SubQuestionNode, SearchGraph as SG
        nodes = [SubQuestionNode(**n) for n in graph_dict["nodes"]]
        sg = SG(nodes=nodes, reasoning=graph_dict.get("reasoning", ""))
        layers = topological_sort(sg)

        print(f"   按 DAG 分 {len(layers)} 层搜索...")
        all_results = []
        for layer_idx, layer in enumerate(layers):
            questions = [n.question for n in layer]
            print(f"   层 {layer_idx}: 并行搜索 {len(questions)} 个子问题...")

            layer_results = _parallel_search(questions, strategy, user_id)
            all_results.extend(layer_results)

            for r in layer_results:
                src = r.get("source_type", "web")
                print(f"   [{src}] {r['sub_question'][:40]}...")

        results = all_results
    else:
        print(f"   并行搜索 {len(sub_questions)} 个子问题...")
        results = _parallel_search(sub_questions, strategy, user_id)
        for r in results:
            src = r.get("source_type", "web")
            print(f"   [{src}] {r['sub_question'][:40]}...")

    scratchpad_entry = (
        f"[Searcher 轮次{current_round} {strategy}] "
        f"搜索了 {len(results)} 个子问题: "
        + " | ".join(r['sub_question'][:30] for r in results)
    )

    return {"search_results": list(results), "scratchpad": [scratchpad_entry]}


def _parallel_search(questions: list[str], strategy: str = "web_only", user_id: str = "default") -> list[dict]:
    """并行搜索多个子问题"""
    from functools import partial

    search_fn = partial(search_and_summarize, strategy=strategy, user_id=user_id)

    async def search_all():
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, search_fn, q)
            for q in questions
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            results = [search_fn(q) for q in questions]
        else:
            results = loop.run_until_complete(search_all())
    except RuntimeError:
        results = asyncio.run(search_all())

    return list(results)


def rag_store_node(state: MindSearchState) -> dict:
    """RAG Store: 把本轮搜索结果存入向量库"""
    memory = get_memory()
    search_results = state.get("search_results", [])

    new_count = len(state.get("sub_questions", []))
    new_results = search_results[-new_count:] if new_count > 0 else []

    docs = [
        Document(
            page_content=r["summary"],
            metadata={
                "sub_question": r["sub_question"],
                "question": state["question"],
            },
        )
        for r in new_results
    ]

    if docs:
        memory.add_documents(docs)
        print(f"\n💾 已将 {len(docs)} 条搜索结果存入本地知识库")

    return {}


def citation_map_node(state: MindSearchState) -> dict:
    """Citation Mapper: 将所有 Searcher 的局部引用统一映射为全局编号"""
    search_results = state.get("search_results", [])

    global_citations, updated_results = _build_global_citations(search_results)

    if global_citations:
        print(f"\n📑 引用映射: {len(global_citations)} 条全局引用")

    scratchpad_entry = f"[Citation Map] 映射了 {len(global_citations)} 条全局引用"

    # search_results 用了 operator.add 不能直接覆盖，写入独立字段 mapped_search_results
    # Synthesizer 优先读 mapped_search_results（带全局引用编号），无则回退 search_results
    return {
        "global_citations": global_citations,
        "mapped_search_results": updated_results,
        "scratchpad": [scratchpad_entry],
    }


def evaluator_node(state: MindSearchState) -> dict:
    """Evaluator: 判断搜索结果是否充分"""
    current_round = state.get("current_round", 1)
    search_results = state.get("search_results", [])

    if current_round >= MAX_ROUNDS:
        print(f"\n⚖️ [轮次 {current_round}] 达到最大轮数 {MAX_ROUNDS}，结束搜索")
        return {"should_continue": False}

    print(f"\n⚖️ [轮次 {current_round}] Evaluator 正在评估搜索结果是否充分...")
    replan = replan_search(state["question"], search_results)

    if replan.need_more and replan.new_sub_questions:
        print(f"   ❌ 信息不足，原因: {replan.reasoning}")
        print(f"   🔄 将追加 {len(replan.new_sub_questions)} 个新问题")
        scratchpad_entry = f"[Evaluator 轮次{current_round}] 信息不足: {replan.reasoning}"
        return {"should_continue": True, "scratchpad": [scratchpad_entry]}
    else:
        print(f"   ✅ 信息充分，原因: {replan.reasoning}")
        scratchpad_entry = f"[Evaluator 轮次{current_round}] 信息充分: {replan.reasoning}"
        return {"should_continue": False, "scratchpad": [scratchpad_entry]}


def synthesizer_node(state: MindSearchState) -> dict:
    """Synthesizer: 汇总所有搜索结果 + 本地知识 + 全局引用 + 用户偏好"""
    memory = get_memory()
    user_id = state.get("user_id", "default")
    ai_mem = get_ai_memory(user_id)
    total_results = len(state.get("search_results", []))
    total_rounds = state.get("current_round", 1)
    local_knowledge = state.get("local_knowledge", [])
    scratchpad = state.get("scratchpad", [])
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)
    global_citations = state.get("global_citations", [])

    # 召回用户偏好
    prefs = ai_mem.recall_preferences(state["question"])
    pref_context = ""
    if prefs:
        pref_lines = [f"- [{p['type']}] {p['text']}" for p in prefs[:5]]
        pref_context = "\n用户偏好（据此调整回答风格和侧重点）:\n" + "\n".join(pref_lines)

    if critique:
        print(f"\n📝 Synthesizer 正在根据批评意见修正回答（第 {revision_count + 1} 次修正）...")
    else:
        print(f"\n📝 Synthesizer 正在汇总（{total_rounds} 轮，{total_results} 条结果，{len(global_citations)} 条引用）...")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    # 优先用 mapped_search_results（summary 中的局部引用已替换为全局编号）；否则回退原始
    results_for_context = state.get("mapped_search_results") or state["search_results"]
    search_context = "\n\n".join(
        f"### 子问题: {r['sub_question']}\n{r['summary']}"
        for r in results_for_context
    )

    # 组装全局引用列表供 LLM 参考
    citation_list = "\n".join(
        f"[{c['global_index']}] {c['title']} — {c['url']}"
        for c in global_citations
    ) if global_citations else "（无引用数据）"

    local_context = "\n".join(
        f"- {item}" for item in local_knowledge
    ) if local_knowledge else "（无历史数据）"

    scratchpad_context = "\n".join(scratchpad) if scratchpad else "（无）"

    critique_section = ""
    if critique:
        critique_section = (
            f"\n\n## 上一版回答的批评意见（必须针对性修正）:\n{critique}\n\n"
            f"## 上一版回答:\n{state.get('final_answer', '')}\n"
        )

    # 命中 skill 时优先使用 skill.system_prompt（保留通用规则作为补充）
    skill_prompt = state.get("skill_system_prompt", "")
    skill_name = state.get("skill_name", "")
    base_system = skill_prompt if skill_prompt else (
        "你是 MindSearch-Lite 智能搜索助手。\n\n"
        "**篇幅控制（必须遵守）:**\n"
        "- 回答控制在 300-500 字以内，精炼为主，不要堆砌\n"
        "- 每个要点 1-2 句话讲清楚即可，不要展开过多\n"
        "- 不要使用「值得注意的是」「总的来说」等套话\n\n"
        "其他要求:\n"
        "1. 优先使用网络搜索结果，本地知识库仅作补充\n"
        "2. 结构化输出，使用标题和列表\n"
        "3. 不同来源有矛盾时指出差异\n"
        "4. 用中文回答"
    )
    if skill_prompt:
        print(f"   ✨ 使用 skill={skill_name} 的专属 system_prompt")

    # 输出格式约束（来自 skill.output_format）
    output_format = state.get("skill_output_format", "") or "markdown"
    format_instruction = {
        "markdown": "**输出格式：Markdown**（使用标题 ## / 列表 - / 加粗 ** / 表格等）",
        "json": "**输出格式：合法 JSON**（顶层对象，包含 summary、key_points、citations 三个键，不要任何额外文本）",
        "plain": "**输出格式：纯文本**（不要使用任何 Markdown 标记，连续段落即可）",
    }.get(output_format, "**输出格式：Markdown**")

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         base_system + "\n\n"
         + format_instruction + "\n\n"
         "**引用规则（必须严格遵守，与 skill 无关）:**\n"
         "- 每个具体事实后紧跟 [n] 引用，n 对应全局引用编号\n"
         "- 引用要精确到具体事实，不要在段落末尾笼统标注\n"
         "- 错误示范: \"美国在AI领域发展迅速 [1][2][3]\"（笼统）\n"
         "- 正确示范: \"OpenAI 发布了 GPT-4 [1]，Google 推出 Gemini [3]\"\n"
         "- 在回答末尾附「参考来源」列表\n\n"
         "**数字与数据规则:**\n"
         "- 只引用搜索结果中明确出现的数字，不要推算或编造\n"
         "- 如果数字来源不确定或多个来源冲突，用\"据报道\"\"约\"等限定词\n"
         "- 没有来源的数字宁可不写"
         + ("\n\n**重要：你正在修正上一版回答，必须针对批评意见逐条改进**" if critique else "")),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human",
         "用户问题: {question}\n\n"
         "思考记录（Scratchpad）:\n{scratchpad}\n\n"
         "网络搜索结果（已带引用编号）:\n{search_context}\n\n"
         "全局引用列表:\n{citation_list}\n\n"
         "本地知识库:\n{local_context}"
         "{pref_context}"
         "{critique_section}\n\n"
         "请综合以上信息，给出精炼回答（300-500字），并在末尾附参考来源。"),
    ])

    parser = StrOutputParser()
    answer = (prompt | llm | parser).invoke({
        "question": state["question"],
        "search_context": search_context,
        "citation_list": citation_list,
        "local_context": local_context,
        "pref_context": pref_context,
        "scratchpad": scratchpad_context,
        "critique_section": critique_section,
        "chat_history": memory.get_chat_history(),
    })

    return {"final_answer": answer, "revision_count": revision_count + 1}


def reflector_node(state: MindSearchState) -> dict:
    """Reflector: 反思 + 自我批评"""
    revision_count = state.get("revision_count", 0)
    final_answer = state.get("final_answer", "")
    scratchpad = state.get("scratchpad", [])

    user_id = state.get("user_id", "default")

    if revision_count >= MAX_REVISIONS:
        print(f"\n🪞 Reflector: 已达最大修正次数 {MAX_REVISIONS}，接受当前回答")
        memory = get_memory()
        memory.add_to_chat_history("human", state["question"])
        memory.add_to_chat_history("ai", final_answer)
        _save_memory_after_accept(state["question"], final_answer, user_id)
        scratchpad_entry = f"[Reflector] 达到最大修正次数，接受回答"
        return {"should_revise": False, "scratchpad": [scratchpad_entry]}

    print(f"\n🪞 Reflector 正在反思和自我批评...")

    # 使用评估模型（与生成模型交叉检查），未配置则回退到主模型
    eval_key = EVAL_API_KEY or OPENAI_API_KEY
    eval_url = EVAL_BASE_URL or OPENAI_BASE_URL
    eval_model = EVAL_MODEL_NAME or MODEL_NAME
    llm = ChatOpenAI(
        model=eval_model,
        api_key=eval_key,
        base_url=eval_url,
    )

    scratchpad_context = "\n".join(scratchpad) if scratchpad else "（无）"
    search_summaries = "\n".join(
        f"- [{r['sub_question']}]: {r['summary'][:100]}..."
        for r in state.get("search_results", [])
    )

    reflect_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个严格的回答质量审查官。你的任务是对 AI 生成的回答进行反思和自我批评。\n\n"
         "你需要从以下维度审查：\n"
         "1. **准确性**: 回答中的事实是否与搜索结果一致？有没有编造信息？\n"
         "2. **完整性**: 是否覆盖了所有子问题？有没有重要信息被遗漏？\n"
         "3. **逻辑性**: 论述是否有逻辑漏洞或自相矛盾？\n"
         "4. **引用质量**: 关键信息是否都有 [n] 引用标注？引用是否准确？\n"
         "5. **偏见检测**: 回答是否存在明显偏见或片面性？\n\n"
         "审查后，给出你的判断：\n"
         "- 如果回答质量足够好（没有重大问题），输出 VERDICT: ACCEPT\n"
         "- 如果有需要修正的问题，输出 VERDICT: REVISE，并给出具体的批评和改进建议\n\n"
         "注意：不要吹毛求疵，只关注真正影响回答质量的问题。"),
        ("human",
         "用户问题: {question}\n\n"
         "思考记录（搜索过程）:\n{scratchpad}\n\n"
         "搜索结果摘要:\n{search_summaries}\n\n"
         "AI 生成的回答:\n{answer}\n\n"
         "请进行反思和自我批评。"),
    ])

    parser = StrOutputParser()
    reflection = (reflect_prompt | llm | parser).invoke({
        "question": state["question"],
        "scratchpad": scratchpad_context,
        "search_summaries": search_summaries,
        "answer": final_answer,
    })

    # Reflector 详细内容只写日志，不暴露给最终用户
    import logging
    logger = logging.getLogger("mindsearch.reflector")
    logger.debug(f"反思详情:\n{reflection}")

    need_revise = "VERDICT: REVISE" in reflection.upper() or "REVISE" in reflection.upper().split("VERDICT")[-1] if "VERDICT" in reflection.upper() else False

    if need_revise:
        print(f"\n🪞 Reflector: 发现可改进点，正在修正...")
        scratchpad_entry = f"[Reflector 第{revision_count + 1}次审查] 需要修正"
        return {
            "should_revise": True,
            "critique": reflection,
            "scratchpad": [scratchpad_entry],
        }
    else:
        print(f"\n🪞 Reflector: 审查通过")
        memory = get_memory()
        memory.add_to_chat_history("human", state["question"])
        memory.add_to_chat_history("ai", final_answer)
        _save_memory_after_accept(state["question"], final_answer, user_id)
        scratchpad_entry = f"[Reflector 第{revision_count + 1}次审查] 通过"
        return {
            "should_revise": False,
            "critique": "",
            "scratchpad": [scratchpad_entry],
        }


def _save_memory_after_accept(question: str, answer: str, user_id: str):
    """回答被接受后异步提取并保存用户偏好"""
    try:
        ai_mem = get_ai_memory(user_id)
        ai_mem.add_turn(question, answer)
        saved = ai_mem.extract_and_save_preferences(question, answer)
        if saved:
            types = [s["content"] for s in saved]
            print(f"   💡 提取到 {len(saved)} 条用户偏好: {', '.join(types)}")
    except Exception as e:
        logger.warning("用户偏好提取/保存失败: %s", e)


# ── 条件路由 ────────────────────────────────────────────────

def should_continue(state: MindSearchState) -> Literal["continue", "finish"]:
    if state.get("should_continue", False):
        return "continue"
    return "finish"


def should_revise(state: MindSearchState) -> Literal["revise", "accept"]:
    if state.get("should_revise", False):
        return "revise"
    return "accept"


# ── 构建 Graph ──────────────────────────────────────────────

def build_mindsearch_graph():
    """构建 MindSearch-Lite 闭环工作流（含 引用系统 + Graph Planner + 两阶段搜索）

    图结构:
      START → planner → rag_retrieve → searcher → rag_store → citation_map → evaluator
                ↑                                                                 |
                └──────────────── "continue" ─────────────────────────────────────┘
                                                                                   |
                                                                            "finish" → synthesizer → reflector
                                                                                          ↑              |
                                                                                          └── "revise" ──┘
                                                                                                          |
                                                                                                   "accept" → END
    """
    graph = StateGraph(MindSearchState)

    # 节点
    graph.add_node("planner", planner_node)
    graph.add_node("rag_retrieve", rag_retrieve_node)
    graph.add_node("searcher", searcher_node)
    graph.add_node("rag_store", rag_store_node)
    graph.add_node("citation_map", citation_map_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("reflector", reflector_node)

    # 边
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "rag_retrieve")
    graph.add_edge("rag_retrieve", "searcher")
    graph.add_edge("searcher", "rag_store")
    graph.add_edge("rag_store", "citation_map")
    graph.add_edge("citation_map", "evaluator")

    # 条件边: 搜索闭环
    graph.add_conditional_edges(
        "evaluator",
        should_continue,
        {
            "continue": "planner",
            "finish": "synthesizer",
        },
    )

    # synthesizer → reflector
    graph.add_edge("synthesizer", "reflector")

    # 条件边: 反思闭环
    graph.add_conditional_edges(
        "reflector",
        should_revise,
        {
            "revise": "synthesizer",
            "accept": END,
        },
    )

    return graph.compile()


def run_mindsearch(
    question: str,
    strategy: str = "web_only",
    user_id: str = "default",
    skill=None,
) -> tuple[str, list[dict]]:
    """运行 MindSearch-Lite

    Args:
        question: 用户查询
        strategy: 搜索策略 (web_only / hybrid)
        user_id: 用户 ID（用于个人文档检索）
        skill: 命中的 Skill 对象（来自 core.skills），用于覆盖 Synthesizer prompt

    Returns:
        (final_answer, global_citations)
        global_citations: list[{global_index, title, url, snippet, sub_question}]
    """
    app = build_mindsearch_graph()

    result = app.invoke({
        "question": question,
        "sub_questions": [],
        "search_graph": None,
        "search_results": [],
        "local_knowledge": [],
        "scratchpad": [],
        "current_round": 0,
        "should_continue": False,
        "final_answer": "",
        "critique": "",
        "revision_count": 0,
        "should_revise": False,
        "global_citations": [],
        "mapped_search_results": [],
        "search_strategy": strategy,
        "user_id": user_id,
        "skill_system_prompt": skill.system_prompt if skill else "",
        "skill_name": skill.name if skill else "",
        "skill_max_sub_questions": skill.max_sub_questions if skill else 0,
        "skill_output_format": skill.output_format if skill else "",
    })

    return result["final_answer"], result.get("global_citations", [])


if __name__ == "__main__":
    answer, citations = run_mindsearch(
        "对比中国和美国在人工智能领域的最新政策和发展情况",
        strategy="web_only",
    )
    print(f"\n{'='*60}")
    print(answer)
    print(f"\n引用: {len(citations)} 条")
