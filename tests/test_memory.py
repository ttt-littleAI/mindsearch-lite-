"""AI Memory：短期对话 / 长期偏好 / 工作记忆"""
import sys
sys.path.insert(0, ".")
from core.memory import get_ai_memory

USER = "test_memory_user"
mem = get_ai_memory(USER)

# 1. 短期记忆
print("=== 短期记忆 ===")
mem.add_turn("什么是RAG", "RAG是检索增强生成...")
mem.add_turn("Transformer怎么工作", "Transformer基于自注意力机制...")
print(f"对话轮次: {len(mem.chat_history) // 2}")
print(f"上下文摘要:\n{mem.get_context_summary()}\n")

# 2. 长期记忆 - 手动保存
print("=== 长期记忆（手动保存） ===")
mem.save_preference("关注大语言模型和RAG技术", memory_type="DOMAIN")
mem.save_preference("喜欢简洁技术风格的回答", memory_type="PREFERENCE")
mem.save_preference("有深度学习基础", memory_type="EXPERTISE")

prefs = mem.recall_preferences("LLM相关技术")
print(f"召回 {len(prefs)} 条偏好:")
for p in prefs:
    print(f"  [{p['type']}] {p['text']} (score: {p['score']:.3f})")

# 3. 长期记忆 - 自动提取
print("\n=== 长期记忆（自动提取） ===")
saved = mem.extract_and_save_preferences(
    question="帮我对比一下PyTorch和TensorFlow在工业部署方面的优劣",
    answer="PyTorch更灵活适合研究，TensorFlow在部署方面有TF Serving等成熟方案..."
)
print(f"自动提取到 {len(saved)} 条偏好:")
for s in saved:
    print(f"  [{s['type']}] {s['content']}")

# 4. 工作记忆
print("\n=== 工作记忆 ===")
mem.start_task("对比中美AI政策")
mem.working.add_finding("美国发布AI行政令")
mem.working.add_finding("中国推出生成式AI管理办法")
mem.working.search_rounds = 2
print(mem.get_working_context())

# 5. 综合召回
print("\n=== 综合召回 ===")
full = mem.recall_all("深度学习框架")
print(f"chat_context: {len(full['chat_context'])} chars")
print(f"preferences: {len(full['preferences'])} items")
print(f"working: {len(full['working'])} chars")

print("\n=== Memory 测试完成 ===")
