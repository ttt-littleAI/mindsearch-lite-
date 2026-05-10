"""Planner Agent — 把复杂问题拆解为子问题 DAG（有向无环图）

输出 graph schema: 每个子问题有 id + dependencies，
支持并行执行无依赖的节点，按拓扑序处理有依赖的节点。
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from config import EVAL_API_KEY, EVAL_BASE_URL, EVAL_MODEL_NAME


# ── Graph Schema 定义 ─────────────────────────────────────────

class SubQuestionNode(BaseModel):
    """子问题节点"""
    id: int = Field(description="子问题编号，从1开始")
    question: str = Field(description="子问题内容")
    dependencies: list[int] = Field(
        default=[],
        description="依赖的子问题编号列表。空列表表示可以直接搜索，非空表示需要先完成依赖项",
    )


class SearchPlan(BaseModel):
    """搜索计划 — 兼容旧接口"""
    sub_questions: list[str] = Field(description="拆解后的子问题列表（扁平列表，兼容旧代码）")
    reasoning: str = Field(description="拆解思路说明")


class SearchGraph(BaseModel):
    """搜索计划 — Graph Schema"""
    nodes: list[SubQuestionNode] = Field(description="子问题节点列表，构成有向无环图")
    reasoning: str = Field(description="拆解思路和依赖关系说明")


class ReplanResult(BaseModel):
    """追加搜索计划"""
    need_more: bool = Field(description="是否还需要继续搜索")
    new_sub_questions: list[str] = Field(default=[], description="需要追加搜索的新子问题")
    reasoning: str = Field(description="判断理由")


# ── Prompts ───────────────────────────────────────────────────

PLANNER_GRAPH_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个问题分析专家。你的任务是把用户的复杂问题拆解成多个子问题，"
     "并建立它们之间的依赖关系，形成一个有向无环图 (DAG)。\n\n"
     "规则：\n"
     "1. 每个子问题应该足够具体，可以直接用搜索引擎搜索\n"
     "2. 如果子问题 B 需要子问题 A 的结果才能搜索，则 B 依赖 A\n"
     "3. 没有依赖关系的子问题可以并行搜索\n"
     "4. **严格限制：最多拆解为 {max_questions} 个子问题**（这是硬上限，超过的会被丢弃）\n"
     "5. 如果问题已经很简单，返回 1 个无依赖的子问题即可\n"
     "6. 子问题用中文表述\n"
     "7. 编号从 1 开始\n\n"
     "示例:\n"
     "问题: \"对比 Tesla 和 BYD 的最新季度财报\"\n"
     "节点:\n"
     "  - id=1, question=\"Tesla 最新季度财报数据\", dependencies=[]\n"
     "  - id=2, question=\"BYD 最新季度财报数据\", dependencies=[]\n"
     "  - id=3, question=\"Tesla 与 BYD 财报对比分析要点\", dependencies=[1, 2]"),
    ("human", "{question}"),
])

DEFAULT_MAX_QUESTIONS = 5


REPLAN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个搜索质量评估专家。\n"
     "用户提了一个问题，已经搜索了一些子问题并得到了结果。\n"
     "你需要判断：当前的搜索结果是否足够回答用户的原始问题？\n\n"
     "判断标准：\n"
     "1. 原始问题的各个方面是否都被覆盖了\n"
     "2. 搜索结果中是否有明显的信息缺口\n"
     "3. 是否有需要深入追问的点\n\n"
     "如果不够，给出需要追加搜索的新子问题（不要重复已搜索的问题）。\n"
     "如果已经足够，设置 need_more 为 false。"),
    ("human",
     "用户原始问题: {question}\n\n"
     "已搜索的子问题和结果:\n{existing_results}\n\n"
     "请判断是否需要追加搜索。"),
])


# ── 核心函数 ──────────────────────────────────────────────────

def _get_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )


def _get_eval_llm():
    """获取评估模型，未配置则回退到主模型"""
    if EVAL_API_KEY and EVAL_BASE_URL and EVAL_MODEL_NAME:
        return ChatOpenAI(
            model=EVAL_MODEL_NAME,
            api_key=EVAL_API_KEY,
            base_url=EVAL_BASE_URL,
        )
    return _get_llm()


def plan_search_graph(question: str, max_questions: int = DEFAULT_MAX_QUESTIONS) -> SearchGraph:
    """对问题进行拆解，返回 Graph Schema（DAG）

    Args:
        question: 用户原始问题
        max_questions: 子问题数量硬上限（来自 skill.max_sub_questions，默认 5）。
                       LLM 会被 prompt 约束在此上限内；超额节点会被截断，
                       同时清理悬空依赖。
    """
    llm = _get_llm()
    chain = PLANNER_GRAPH_PROMPT | llm.with_structured_output(SearchGraph, method="function_calling")
    graph = chain.invoke({"question": question, "max_questions": max_questions})

    # 兜底截断：LLM 不听话超额时，保留前 max_questions 个节点
    if len(graph.nodes) > max_questions:
        graph.nodes = graph.nodes[:max_questions]

    # 校验：确保没有循环依赖、依赖编号合法（被截断的 id 也要从依赖里清理）
    valid_ids = {n.id for n in graph.nodes}
    for node in graph.nodes:
        node.dependencies = [d for d in node.dependencies if d in valid_ids and d != node.id]

    return graph


def plan_search(question: str) -> SearchPlan:
    """对问题进行拆解（兼容旧接口，内部调用 graph 版本再展平）"""
    graph = plan_search_graph(question)
    return SearchPlan(
        sub_questions=[n.question for n in graph.nodes],
        reasoning=graph.reasoning,
    )


def topological_sort(graph: SearchGraph) -> list[list[SubQuestionNode]]:
    """将 DAG 按拓扑序分层，同一层的节点可以并行执行。

    返回: [[layer0_nodes], [layer1_nodes], ...]
    """
    nodes_by_id = {n.id: n for n in graph.nodes}
    in_degree = {n.id: len(n.dependencies) for n in graph.nodes}
    resolved = set()
    layers = []

    while len(resolved) < len(graph.nodes):
        # 找出入度为 0 的节点（所有依赖已完成）
        layer = [
            nodes_by_id[nid]
            for nid, deg in in_degree.items()
            if deg == 0 and nid not in resolved
        ]
        if not layer:
            # 有环或孤立节点，把剩余的全放一层
            layer = [nodes_by_id[nid] for nid in in_degree if nid not in resolved]

        layers.append(layer)
        for n in layer:
            resolved.add(n.id)
        # 更新入度
        for nid in in_degree:
            if nid not in resolved:
                in_degree[nid] = sum(
                    1 for d in nodes_by_id[nid].dependencies if d not in resolved
                )

    return layers


def replan_search(question: str, search_results: list[dict]) -> ReplanResult:
    """根据已有搜索结果判断是否需要追加搜索（使用评估模型）"""
    llm = _get_eval_llm()

    existing_results = "\n\n".join(
        f"### 子问题: {r['sub_question']}\n摘要: {r['summary']}"
        for r in search_results
    )

    chain = REPLAN_PROMPT | llm.with_structured_output(ReplanResult, method="function_calling")
    return chain.invoke({
        "question": question,
        "existing_results": existing_results,
    })


if __name__ == "__main__":
    graph = plan_search_graph("对比中国和美国在人工智能领域的最新政策和发展情况")
    print(f"拆解思路: {graph.reasoning}")
    layers = topological_sort(graph)
    for layer_idx, layer in enumerate(layers):
        print(f"\n第 {layer_idx + 1} 层（可并行）:")
        for n in layer:
            deps = f" ← 依赖 {n.dependencies}" if n.dependencies else ""
            print(f"  [{n.id}] {n.question}{deps}")
