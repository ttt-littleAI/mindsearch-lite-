"""RAGAS 评估框架 — 用独立模型评估搜索回答质量

评估维度:
  faithfulness    — 回答是否忠于搜索结果（不编造）
  answer_relevancy — 回答是否切题
  context_precision — 检索到的上下文是否精准
  context_recall   — 检索是否覆盖了所有必要信息

评估模型: Qwen-plus（阿里云百炼），与生成模型 DeepSeek 交叉检查
"""

from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import EVAL_API_KEY, EVAL_BASE_URL, EVAL_MODEL_NAME
from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

import re


@dataclass
class EvalResult:
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall: float = 0.0
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (
            f"faithfulness={self.faithfulness:.2f} "
            f"relevancy={self.answer_relevancy:.2f} "
            f"precision={self.context_precision:.2f} "
            f"recall={self.context_recall:.2f} "
            f"overall={self.overall:.2f}"
        )


def _get_eval_llm():
    """获取评估模型（优先 Qwen，回退 DeepSeek）"""
    api_key = EVAL_API_KEY if EVAL_API_KEY and EVAL_API_KEY != "你的阿里云百炼API密钥" else OPENAI_API_KEY
    base_url = EVAL_BASE_URL if EVAL_BASE_URL else OPENAI_BASE_URL
    model = EVAL_MODEL_NAME if EVAL_MODEL_NAME else MODEL_NAME

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0,
    )


EVAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个搜索回答质量评估专家。请从以下4个维度评估AI的回答质量。\n\n"
     "每个维度打分 0.0-1.0：\n"
     "1. **FAITHFULNESS** — 回答是否忠于上下文/搜索结果？有无编造信息？\n"
     "   1.0=完全忠实 0.0=大量编造\n"
     "2. **ANSWER_RELEVANCY** — 回答是否切中问题？有无答非所问？\n"
     "   1.0=完全切题 0.0=完全跑题\n"
     "3. **CONTEXT_PRECISION** — 检索到的上下文是否精准相关？有无噪音？\n"
     "   1.0=全部相关 0.0=全是噪音\n"
     "4. **CONTEXT_RECALL** — 回答是否利用了所有相关上下文？有无遗漏？\n"
     "   1.0=充分利用 0.0=大量遗漏\n\n"
     "严格按以下格式输出（4行，每行一个分数）：\n"
     "FAITHFULNESS: 0.X\n"
     "ANSWER_RELEVANCY: 0.X\n"
     "CONTEXT_PRECISION: 0.X\n"
     "CONTEXT_RECALL: 0.X"),
    ("human",
     "用户问题: {question}\n\n"
     "检索到的上下文:\n{context}\n\n"
     "AI回答:\n{answer}\n\n"
     "请评分:"),
])

_SCORE_PATTERN = re.compile(r"(FAITHFULNESS|ANSWER_RELEVANCY|CONTEXT_PRECISION|CONTEXT_RECALL):\s*([\d.]+)", re.IGNORECASE)


def evaluate(question: str, answer: str, contexts: list[str]) -> EvalResult:
    """评估一次搜索回答的质量"""
    llm = _get_eval_llm()
    context_text = "\n---\n".join(contexts) if contexts else "(无上下文)"

    try:
        raw = (EVAL_PROMPT | llm | StrOutputParser()).invoke({
            "question": question,
            "context": context_text,
            "answer": answer,
        })
        return _parse_scores(raw)
    except Exception as e:
        return EvalResult(details={"error": str(e)})


def _parse_scores(text: str) -> EvalResult:
    scores = {}
    for match in _SCORE_PATTERN.finditer(text):
        key = match.group(1).lower()
        try:
            scores[key] = min(1.0, max(0.0, float(match.group(2))))
        except ValueError:
            scores[key] = 0.0

    result = EvalResult(
        faithfulness=scores.get("faithfulness", 0.0),
        answer_relevancy=scores.get("answer_relevancy", 0.0),
        context_precision=scores.get("context_precision", 0.0),
        context_recall=scores.get("context_recall", 0.0),
    )
    result.overall = (
        result.faithfulness * 0.3
        + result.answer_relevancy * 0.3
        + result.context_precision * 0.2
        + result.context_recall * 0.2
    )
    return result


def batch_evaluate(test_cases: list[dict]) -> list[EvalResult]:
    """批量评估

    test_cases: [{"question": str, "answer": str, "contexts": list[str]}, ...]
    """
    return [
        evaluate(tc["question"], tc["answer"], tc.get("contexts", []))
        for tc in test_cases
    ]


def print_eval_report(results: list[EvalResult], test_cases: list[dict]):
    """打印评估报告"""
    print(f"\n{'='*60}")
    print(f"RAGAS 评估报告 — {len(results)} 条测试")
    print(f"{'='*60}")

    for i, (result, tc) in enumerate(zip(results, test_cases)):
        print(f"\n[{i+1}] {tc['question'][:50]}")
        print(f"    {result.summary()}")

    if results:
        avg = EvalResult(
            faithfulness=sum(r.faithfulness for r in results) / len(results),
            answer_relevancy=sum(r.answer_relevancy for r in results) / len(results),
            context_precision=sum(r.context_precision for r in results) / len(results),
            context_recall=sum(r.context_recall for r in results) / len(results),
        )
        avg.overall = (avg.faithfulness * 0.3 + avg.answer_relevancy * 0.3
                       + avg.context_precision * 0.2 + avg.context_recall * 0.2)
        print(f"\n{'─'*60}")
        print(f"平均: {avg.summary()}")
        print(f"{'='*60}")
