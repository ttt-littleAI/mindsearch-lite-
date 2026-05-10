"""路由器准确率测试 — 50 条标注数据"""
import sys
import os
import re
import requests

sys.path.insert(0, ".")
from config.settings import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

ROUTER_SYSTEM = (
    "你是一个查询意图分类器。根据用户的问题，判断它属于以下哪个类别。\n\n"
    "**REALTIME** — 需要最新、实时的信息才能回答\n"
    "  例: 今天的天气、最新的股价、刚发布的新闻、当前的政策\n"
    "  信号词: 今天、最新、刚刚、目前、现在、最近（一周内）\n\n"
    "**STABLE** — 稳定的知识，不会频繁变化\n"
    "  例: 什么是机器学习、Python和Java的区别、光合作用的原理\n"
    "  信号词: 什么是、如何、为什么、对比、原理、概念\n\n"
    "**PERSONAL** — 涉及用户自己的文档或私有数据\n"
    "  例: 我上传的论文里提到了什么、我的报告中的数据、我的文档\n"
    "  信号词: 我的、我上传的、我的文档、我的文件\n\n"
    "严格按以下格式输出（三行，不要多余内容）:\n"
    "CATEGORY: REALTIME 或 STABLE 或 PERSONAL\n"
    "REASON: 一句话理由\n"
    "CONFIDENCE: 0到1之间的数字"
)

_CAT_PAT = re.compile(r"CATEGORY:\s*(REALTIME|STABLE|PERSONAL)", re.IGNORECASE)
_CONF_PAT = re.compile(r"CONFIDENCE:\s*([\d.]+)", re.IGNORECASE)


def classify_query(question: str):
    resp = requests.post(
        f"{OPENAI_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": ROUTER_SYSTEM},
                {"role": "user", "content": f"问题: {question}"},
            ],
            "temperature": 0,
        },
        timeout=30,
    )
    text = resp.json()["choices"][0]["message"]["content"]
    cat_m = _CAT_PAT.search(text)
    conf_m = _CONF_PAT.search(text)
    category = cat_m.group(1).upper() if cat_m else "REALTIME"
    try:
        confidence = float(conf_m.group(1)) if conf_m else 0.5
    except ValueError:
        confidence = 0.5
    return type("R", (), {"category": category, "confidence": confidence})()

TEST_CASES = [
    # ── REALTIME（20条）──
    ("今天A股大盘走势如何", "REALTIME"),
    ("最新的ChatGPT更新了什么功能", "REALTIME"),
    ("今天北京天气怎么样", "REALTIME"),
    ("刚刚发布的iPhone 17有什么亮点", "REALTIME"),
    ("目前俄乌局势最新进展", "REALTIME"),
    ("现在比特币价格是多少", "REALTIME"),
    ("今天有什么AI新闻", "REALTIME"),
    ("最近一周OpenAI有什么动态", "REALTIME"),
    ("今年高考分数线出来了吗", "REALTIME"),
    ("当前国内油价多少钱一升", "REALTIME"),
    ("最新的新能源汽车销量排行", "REALTIME"),
    ("今天的热搜是什么", "REALTIME"),
    ("刚发布的论文讲了什么", "REALTIME"),
    ("现在去日本旅游签证政策是什么", "REALTIME"),
    ("最近有什么好看的电影上映", "REALTIME"),
    ("今天的NBA比赛结果", "REALTIME"),
    ("目前国内大模型有哪些最新发布", "REALTIME"),
    ("近期人民币汇率走势如何", "REALTIME"),
    ("最新的考研政策变化", "REALTIME"),
    ("现在深圳的房价是多少", "REALTIME"),

    # ── STABLE（20条）──
    ("什么是Transformer的自注意力机制", "STABLE"),
    ("Python和Java的区别是什么", "STABLE"),
    ("机器学习和深度学习有什么不同", "STABLE"),
    ("光合作用的原理是什么", "STABLE"),
    ("如何用PyTorch搭建一个CNN", "STABLE"),
    ("BERT模型的架构是怎样的", "STABLE"),
    ("什么是梯度下降算法", "STABLE"),
    ("TCP和UDP的区别", "STABLE"),
    ("数据库索引的原理是什么", "STABLE"),
    ("什么是微服务架构", "STABLE"),
    ("ResNet为什么能训练很深的网络", "STABLE"),
    ("什么是注意力机制", "STABLE"),
    ("Docker和虚拟机的区别", "STABLE"),
    ("如何理解反向传播算法", "STABLE"),
    ("Linux常用命令有哪些", "STABLE"),
    ("什么是RAG检索增强生成", "STABLE"),
    ("Redis和MySQL的使用场景区别", "STABLE"),
    ("BM25算法的原理是什么", "STABLE"),
    ("GAN的生成器和判别器怎么训练", "STABLE"),
    ("如何用LangChain构建Agent", "STABLE"),

    # ── PERSONAL（10条）──
    ("我上传的论文讲了什么内容", "PERSONAL"),
    ("我的文档中提到了哪些实验结果", "PERSONAL"),
    ("帮我总结一下我上传的报告", "PERSONAL"),
    ("我的文件里关于深度学习的部分在哪", "PERSONAL"),
    ("我之前上传的简历有什么问题", "PERSONAL"),
    ("我的笔记中关于Transformer的内容", "PERSONAL"),
    ("我上传的PDF第三页讲了什么", "PERSONAL"),
    ("我的文档中有没有提到数据增强", "PERSONAL"),
    ("帮我在我的文件中找关于损失函数的描述", "PERSONAL"),
    ("我上传的代码有什么bug", "PERSONAL"),

    # ── 边界模糊 case（10条）──
    ("Transformer最新的改进有哪些", "REALTIME"),
    ("深度学习的发展历程", "STABLE"),
    ("帮我查一下这个问题", "STABLE"),
    ("怎么看待人工智能的未来", "STABLE"),
    ("分析一下这段代码的性能", "PERSONAL"),
    ("有没有什么好用的Python库推荐", "STABLE"),
    ("大模型幻觉问题怎么解决", "STABLE"),
    ("谁发明了Transformer", "STABLE"),
    ("总结一下最近的AI进展", "REALTIME"),
    ("对比一下GPT-4o和Claude的最新能力", "REALTIME"),
]

print(f"路由准确率测试 — 共 {len(TEST_CASES)} 条\n")
print("=" * 70)

correct = 0
total = len(TEST_CASES)
errors = []

for i, (question, expected) in enumerate(TEST_CASES, 1):
    result = classify_query(question)
    predicted = result.category.upper()
    is_correct = predicted == expected
    if is_correct:
        correct += 1
        status = "✓"
    else:
        status = "✗"
        errors.append((question, expected, predicted, result.confidence))

    print(f"  [{i:02d}] {status} | {predicted:10s} (期望 {expected:10s}) | 置信度 {result.confidence:.0%} | {question[:30]}")

print("=" * 70)
accuracy = correct / total * 100
print(f"\n准确率: {correct}/{total} = {accuracy:.1f}%")

if errors:
    print(f"\n错误 {len(errors)} 条:")
    for q, exp, pred, conf in errors:
        print(f"  期望 {exp} 预测 {pred} (置信度 {conf:.0%}): {q}")

# 分类别统计
for cat in ["REALTIME", "STABLE", "PERSONAL"]:
    cat_cases = [(q, e) for q, e in TEST_CASES if e == cat]
    cat_correct = sum(1 for q, e in cat_cases if classify_query(q).category.upper() == e)
    print(f"  {cat}: {cat_correct}/{len(cat_cases)} = {cat_correct/len(cat_cases)*100:.0f}%")
