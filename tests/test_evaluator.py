"""RAGAS 四维评估框架：faithfulness / answer_relevancy / context_precision / context_recall"""
import sys
sys.path.insert(0, ".")
from core.evaluator import evaluate, batch_evaluate, print_eval_report

# 测试用例
test_cases = [
    {
        "question": "什么是RAG技术？",
        "answer": "RAG（Retrieval-Augmented Generation）是一种将检索和生成结合的技术[1]。它先从知识库中检索相关文档，再将检索结果作为上下文输入给大语言模型生成回答[2]。优点是能减少幻觉、利用最新信息[1]。",
        "contexts": [
            "[1] RAG是检索增强生成技术，通过先检索后生成的方式提升LLM回答质量",
            "[2] RAG的工作流程：用户提问→检索器从知识库检索→将检索结果和问题一起输入LLM→生成回答",
        ],
    },
    {
        "question": "Python和Java哪个更快？",
        "answer": "总的来说，Java通常比Python运行速度快10-100倍[1]。Java是编译型语言，有JIT优化[2]。但Python在AI/ML领域更流行，配合NumPy等C扩展性能也不差。",
        "contexts": [
            "[1] 基准测试显示Java在大多数计算密集型任务中比Python快10-100倍",
            "[2] Java的HotSpot JVM提供JIT编译优化，将热点代码编译为机器码",
        ],
    },
    {
        "question": "今天天气怎么样？",
        "answer": "北京今天晴，气温25度，适合出行。上海多云转阴，气温22度。",
        "contexts": [
            "搜索结果显示各地天气预报数据",
        ],
    },
]

print("=== RAGAS 评估测试 ===\n")

# 单条评估
print("--- 单条评估 ---")
result = evaluate(
    test_cases[0]["question"],
    test_cases[0]["answer"],
    test_cases[0]["contexts"],
)
print(f"Q: {test_cases[0]['question']}")
print(f"Score: {result.summary()}")

# 批量评估
print("\n--- 批量评估 ---")
results = batch_evaluate(test_cases)
print_eval_report(results, test_cases)

print("\n=== Evaluator 测试完成 ===")
