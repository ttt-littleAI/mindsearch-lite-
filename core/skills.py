"""Skills 技能系统 — 预定义搜索技能模板

每个 Skill 封装一套 Prompt + 搜索策略 + 后处理逻辑，
Agent 根据用户意图自动选择最匹配的 Skill 执行。

内置技能:
  academic_search   — 学术搜索（论文、研究、技术原理）
  news_briefing     — 新闻简报（热点事件、最新动态）
  compare_analysis  — 对比分析（多个方案/技术/产品对比）
  document_qa       — 文档问答（基于上传文档回答）
  summarize         — 内容摘要（长文/多源信息提炼）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Skill:
    name: str
    display_name: str
    description: str
    strategy: str
    system_prompt: str
    output_format: str = "markdown"
    max_sub_questions: int = 3
    keywords: list[str] = field(default_factory=list)
    # 正则模式：捕获命令式/格式化输入（"翻译 xxx"、URL、"算 1+1"）。
    # 命中正则视为强信号，每条 +2 分（高于关键词的 +1）。
    patterns: list[str] = field(default_factory=list)


SKILL_REGISTRY: dict[str, Skill] = {}


def register_skill(skill: Skill):
    SKILL_REGISTRY[skill.name] = skill


def get_skill(name: str) -> Skill | None:
    return SKILL_REGISTRY.get(name)


def match_skill(question: str) -> Skill | None:
    """根据问题关键词 + 正则模式匹配最合适的技能"""
    question_lower = question.lower()
    best_match = None
    best_score = 0
    for skill in SKILL_REGISTRY.values():
        score = sum(1 for kw in skill.keywords if kw in question_lower)
        score += sum(2 for p in skill.patterns if re.search(p, question, re.IGNORECASE))
        if score > best_score:
            best_score = score
            best_match = skill
    return best_match if best_score > 0 else None


def list_skills() -> list[dict]:
    return [
        {
            "name": s.name,
            "display_name": s.display_name,
            "description": s.description,
            "strategy": s.strategy,
        }
        for s in SKILL_REGISTRY.values()
    ]


# ── 注册内置技能 ──

register_skill(Skill(
    name="academic_search",
    display_name="学术搜索",
    description="搜索学术论文、技术原理、算法解释，输出结构化的技术分析",
    strategy="hybrid",
    system_prompt=(
        "你是一个学术研究助手。请基于搜索结果给出严谨的技术分析：\n"
        "1. 核心概念解释（用通俗语言）\n"
        "2. 关键技术细节（公式、架构、参数）\n"
        "3. 相关工作对比\n"
        "4. 每个事实标注引用 [n]\n"
        "控制在 300 字以内。"
    ),
    max_sub_questions=4,
    keywords=["论文", "算法", "原理", "模型", "架构", "机制", "方法",
              "transformer", "attention", "cnn", "rnn", "bert", "gpt",
              "训练", "微调", "损失函数", "梯度", "优化"],
))

register_skill(Skill(
    name="news_briefing",
    display_name="新闻简报",
    description="获取最新新闻动态，生成结构化简报",
    strategy="web_only",
    system_prompt=(
        "你是一个新闻编辑。请基于搜索结果生成简报：\n"
        "1. 一句话核心摘要\n"
        "2. 关键事实（时间、人物、数据）\n"
        "3. 影响与后续展望\n"
        "每个事实标注引用 [n]，控制在 200 字以内。"
    ),
    max_sub_questions=2,
    keywords=["新闻", "最新", "今天", "刚刚", "发布", "宣布", "热点",
              "事件", "动态", "进展", "突发", "最近"],
    patterns=[r"^.{1,15}的?新闻", r"今日\S{1,10}头条", r"\d{4}年.*事件"],
))

register_skill(Skill(
    name="compare_analysis",
    display_name="对比分析",
    description="多个方案/技术/产品的结构化对比",
    strategy="hybrid",
    system_prompt=(
        "你是一个分析师。请对搜索结果进行结构化对比：\n"
        "1. 用表格列出各项对比维度\n"
        "2. 各自的优势和劣势\n"
        "3. 适用场景建议\n"
        "4. 总结推荐\n"
        "每个事实标注引用 [n]，控制在 400 字以内。"
    ),
    max_sub_questions=5,
    keywords=["对比", "比较", "区别", "优劣", "哪个好", "vs", "选择",
              "差异", "异同", "推荐"],
    patterns=[r"\S+\s*vs\.?\s*\S+", r"\S+和\S+(的)?(区别|差异|对比)", r"\S+与\S+(的)?(异同|对比)"],
))

register_skill(Skill(
    name="document_qa",
    display_name="文档问答",
    description="基于用户上传的文档回答问题",
    strategy="local_only",
    system_prompt=(
        "你是文档分析助手。请严格基于提供的文档内容回答：\n"
        "1. 直接回答问题\n"
        "2. 引用文档中的原文支持\n"
        "3. 如果文档中没有相关内容，明确说明\n"
        "每个引用标注来源 [n]，控制在 200 字以内。"
    ),
    max_sub_questions=1,
    keywords=["我的文档", "我上传的", "我的文件", "我的论文", "我的报告",
              "文档里", "文件中", "报告中", "上传的", "我的"],
    patterns=[r"我的\S{1,8}(里|中|上)?(说|提到|写)", r"上传的\S{0,8}(文件|文档|资料)"],
))

register_skill(Skill(
    name="summarize",
    display_name="内容摘要",
    description="对长文或多源信息进行提炼总结",
    strategy="hybrid",
    system_prompt=(
        "你是一个摘要专家。请对搜索结果进行精炼总结：\n"
        "1. 核心观点提取（3-5 条要点）\n"
        "2. 关键数据和结论\n"
        "3. 一句话总结\n"
        "每个要点标注引用 [n]，控制在 250 字以内。"
    ),
    max_sub_questions=3,
    keywords=["总结", "摘要", "概括", "提炼", "归纳", "综述", "概述",
              "简述", "介绍一下", "讲讲"],
))
