"""Phase 4 测试: 时效性路由器"""
import sys
sys.path.insert(0, ".")
from core.router import classify_query

test_cases = [
    ("今天A股大盘走势如何", "REALTIME"),
    ("最新的GPT-5发布了吗", "REALTIME"),
    ("什么是Transformer架构", "STABLE"),
    ("Python和Go语言的性能对比", "STABLE"),
    ("我上传的论文里关于attention机制怎么说的", "PERSONAL"),
    ("我的文档中有哪些关于深度学习的内容", "PERSONAL"),
]

print("=== 时效性路由测试 ===\n")
correct = 0
for question, expected in test_cases:
    result = classify_query(question)
    match = result.category == expected
    correct += match
    status = "OK" if match else "MISS"
    print(f"[{status}] {question}")
    print(f"       expect={expected} got={result.category} conf={result.confidence:.0%}")
    print(f"       reason: {result.reasoning}\n")

print(f"=== {correct}/{len(test_cases)} 正确 ===")
