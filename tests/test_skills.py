"""Skills 技能匹配测试"""
import sys
import importlib.util

sys.path.insert(0, ".")

spec = importlib.util.spec_from_file_location("skills", "core/skills.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["skills"] = mod
spec.loader.exec_module(mod)

match_skill = mod.match_skill
list_skills = mod.list_skills

print("=== 已注册技能 ===")
for s in list_skills():
    print(f"  [{s['name']}] {s['display_name']} — {s['strategy']}")

print("\n=== 匹配测试 ===")
test_cases = [
    "Transformer的自注意力机制原理是什么",
    "今天有什么AI新闻",
    "PyTorch和TensorFlow哪个好",
    "我上传的论文讲了什么",
    "总结一下深度学习的发展",
    "北京天气怎么样",
    "BERT和GPT的区别是什么",
]

for q in test_cases:
    skill = match_skill(q)
    name = f"{skill.display_name}({skill.strategy})" if skill else "无匹配(默认路由)"
    print(f"  Q: {q}")
    print(f"  → {name}\n")
