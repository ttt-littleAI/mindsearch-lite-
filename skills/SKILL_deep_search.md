---
name: deep_search
description: 多Agent闭环深度搜索，适合复杂问题
tools:
  - deep_search
---

# Deep Search — 多Agent深度搜索

## 什么时候用

- 用户问了一个**复杂问题**，需要从多个角度搜索
- 对比类问题（"A 和 B 有什么区别"）
- 需要综合多个来源的信息
- 简单搜索不够用的时候

## 什么时候不用

- 简单事实性问题（"今天天气"、"某人生日"）→ 用 `quick_search`
- 查之前搜过的内容 → 用 `knowledge_query`

## 工作原理

1. **Planner** 把问题拆解成 2-5 个子问题
2. **RAG Retriever** 先查本地知识库有没有相关历史数据
3. **Searcher** 并行搜索所有子问题
4. **RAG Store** 把搜索结果存入本地知识库
5. **Evaluator** 判断结果够不够
   - 不够 → 回到 Planner 追加新子问题，最多 3 轮
   - 够了 → 进入 Synthesizer 汇总
6. **Synthesizer** 结合搜索结果 + 本地知识 + 对话记忆，生成最终回答

## 示例

```
输入: "对比 2024 年 OpenAI 和 Anthropic 发布的主要模型"

Planner 拆解:
  1. 2024年 OpenAI 发布了哪些模型
  2. 2024年 Anthropic 发布了哪些模型
  3. 两家模型的性能对比和评测结果

→ 并行搜索 → 评估 → 可能追加搜索 → 汇总回答
```
