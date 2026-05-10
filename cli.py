"""MindSearch v2 CLI — 命令行交互模式"""

import sys
from agents.react_agent import run_react_agent
from chains.search_chain import run_search_chain
from core.search_engine import search as core_search, SearchRequest


def _run_v2_search(question: str) -> str:
    result = core_search(SearchRequest(question=question))
    return result.answer


MODES = {
    "1": ("基础搜索链", run_search_chain),
    "2": ("ReAct Agent", run_react_agent),
    "3": ("多Agent搜索 (MindSearch v2)", _run_v2_search),
}


def main():
    print("=" * 60)
    print("  🔍 MindSearch-Lite — 多Agent智能搜索引擎")
    print("=" * 60)
    print("\n选择搜索模式:")
    print("  1. 基础搜索链 (Phase 1)")
    print("  2. ReAct Agent (Phase 2)")
    print("  3. 多Agent搜索 (Phase 3 - 推荐)")
    print()

    mode = input("请选择 [1/2/3，默认3]: ").strip() or "3"
    if mode not in MODES:
        print("无效选择")
        sys.exit(1)

    mode_name, run_fn = MODES[mode]
    print(f"\n✅ 已选择: {mode_name}")
    print("输入 'quit' 退出\n")

    while True:
        question = input("🔍 请输入问题: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("再见！")
            break
        if not question:
            continue

        print()
        result = run_fn(question)

        # run_search_chain 返回 dict，其他返回 str
        if isinstance(result, dict):
            print(f"\n{'='*60}")
            print(result["answer"])
        else:
            print(f"\n{'='*60}")
            print(result)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
