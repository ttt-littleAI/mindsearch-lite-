"""零外部依赖的纯逻辑单元测试 — 在 CI 中跑（无需 LLM/Milvus/Redis/MySQL）。

覆盖：
  - core.router 规则前置短路（_PERSONAL_RE / _REALTIME_RE 正则 + classify_query 命中分支）
  - core.skills.match_skill 关键词 + 正则匹配
  - core.chunker._find_semantic_breaks z-score 自适应法
  - core.chunker._split_into_sentences 中英文句子切分
  - core.chunker._cosine 余弦相似度

跑法:  pytest tests/test_pure_logic.py -v
"""
import math

import pytest


# ─────────────────────────────────────────────────────────────
# Router: 规则前置短路（不调 LLM）
# ─────────────────────────────────────────────────────────────

class TestRouterRulePrefix:
    """规则前置应在不调 LLM 的情况下直接返回路由决策。"""

    def test_personal_pattern_matches_my_doc(self):
        from core.router import _PERSONAL_RE
        assert _PERSONAL_RE.search("我的文档里写了什么")
        assert _PERSONAL_RE.search("我上传的报告")
        assert _PERSONAL_RE.search("文档里提到的方法")

    def test_personal_pattern_rejects_unrelated(self):
        from core.router import _PERSONAL_RE
        assert not _PERSONAL_RE.search("什么是 Transformer")
        assert not _PERSONAL_RE.search("今天天气怎么样")

    def test_realtime_pattern_matches_temporal_words(self):
        from core.router import _REALTIME_RE
        assert _REALTIME_RE.search("今天 A 股大盘怎么样")
        assert _REALTIME_RE.search("最新的 GPT-5 发布消息")
        assert _REALTIME_RE.search("最近 OpenAI 的动态")
        assert _REALTIME_RE.search("现在比特币价格多少")

    def test_realtime_pattern_rejects_stable_questions(self):
        from core.router import _REALTIME_RE
        assert not _REALTIME_RE.search("什么是机器学习")
        assert not _REALTIME_RE.search("Python 和 Java 的区别")

    def test_classify_query_personal_shortcircuit(self):
        """规则命中 PERSONAL：不应调 LLM，直接返回 confidence=1.0"""
        from core.router import classify_query
        result = classify_query("我的文档里有哪些深度学习的内容")
        assert result.category == "PERSONAL"
        assert result.confidence == 1.0
        assert "规则命中" in result.reasoning

    def test_classify_query_realtime_shortcircuit(self):
        """规则命中 REALTIME：不应调 LLM，直接返回 confidence=0.95"""
        from core.router import classify_query
        result = classify_query("最新的 GPT-5 有什么消息")
        assert result.category == "REALTIME"
        assert result.confidence == 0.95
        assert "规则命中" in result.reasoning


# ─────────────────────────────────────────────────────────────
# Skills: 关键词 + 正则匹配
# ─────────────────────────────────────────────────────────────

class TestSkillMatching:
    def test_keyword_only_match(self):
        from core.skills import match_skill
        skill = match_skill("帮我对比一下 Python 和 Java")
        assert skill is not None
        assert skill.name == "compare_analysis"

    def test_news_keyword_match(self):
        from core.skills import match_skill
        skill = match_skill("最新的 OpenAI 新闻")
        assert skill is not None
        assert skill.name == "news_briefing"

    def test_pattern_boosts_match_score(self):
        """正则 patterns 命中权重 +2，应优先于纯关键词命中"""
        from core.skills import match_skill
        # "Tesla vs BYD" 命中 compare_analysis 的 patterns（vs 模式）
        skill = match_skill("Tesla vs BYD 财报")
        assert skill is not None
        assert skill.name == "compare_analysis"

    def test_no_match_returns_none(self):
        from core.skills import match_skill
        # 设计成既不匹配 keyword 也不匹配 pattern 的句子
        assert match_skill("xyzzy plugh quux") is None

    def test_document_qa_pattern(self):
        from core.skills import match_skill
        skill = match_skill("我上传的论文文件里说了什么")
        assert skill is not None
        assert skill.name == "document_qa"


# ─────────────────────────────────────────────────────────────
# Chunker: z-score 自适应语义断点
# ─────────────────────────────────────────────────────────────

class TestSemanticBreaks:
    def test_uniform_similarities_no_breaks(self):
        """所有相邻相似度相同 → 没有"反常低值" → 0 个断点"""
        from core.chunker import _find_semantic_breaks
        sims = [0.85, 0.85, 0.85, 0.85, 0.85]
        assert _find_semantic_breaks(sims) == set()

    def test_outlier_low_similarity_detected(self):
        """中间一处相似度异常低（>1σ 偏离）→ 应被识别为断点"""
        from core.chunker import _find_semantic_breaks
        # mean ~ 0.74, std ~ 0.20, 0.10 的 z-score ≈ -3.2 远低于 -1.0
        sims = [0.85, 0.88, 0.10, 0.83, 0.86]
        breaks = _find_semantic_breaks(sims)
        assert 2 in breaks

    def test_too_few_similarities_no_breaks(self):
        """少于 2 个相似度无法做统计"""
        from core.chunker import _find_semantic_breaks
        assert _find_semantic_breaks([]) == set()
        assert _find_semantic_breaks([0.5]) == set()

    def test_zero_variance_no_breaks(self):
        """方差为 0（完全一致）应直接返回空集，不触发除零"""
        from core.chunker import _find_semantic_breaks
        assert _find_semantic_breaks([0.7, 0.7, 0.7]) == set()


class TestSentenceSplit:
    def test_chinese_punctuation(self):
        from core.chunker import _split_into_sentences
        sents = _split_into_sentences("第一句。第二句！第三句？")
        assert sents == ["第一句。", "第二句！", "第三句？"]

    def test_english_period(self):
        from core.chunker import _split_into_sentences
        sents = _split_into_sentences("First sentence. Second one. Third.")
        # 英文 ". " 后跟大写字母才切（避免误切小数点 / 缩写）
        assert "First sentence." in sents

    def test_mixed_zh_en(self):
        """中文句号 + 中文感叹号都应切；英文 '.' 后跟中文不切（避免误切小数点/缩写）"""
        from core.chunker import _split_into_sentences
        sents = _split_into_sentences("中文一。This is English. 中文二！End.")
        # 至少切出 3 段：中文一、This 段（含混合）、End.
        assert len(sents) >= 3
        assert any("中文一" in s for s in sents)
        assert any("End" in s for s in sents)

    def test_empty_input(self):
        from core.chunker import _split_into_sentences
        assert _split_into_sentences("") == []
        assert _split_into_sentences("   ") == []


class TestCosine:
    def test_identical_vectors(self):
        from core.chunker import _cosine
        v = [1.0, 2.0, 3.0]
        assert _cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        from core.chunker import _cosine
        assert _cosine([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_zero_vector_safe(self):
        """0 向量应返回 0 而不是除零异常"""
        from core.chunker import _cosine
        assert _cosine([0, 0, 0], [1, 2, 3]) == 0.0
