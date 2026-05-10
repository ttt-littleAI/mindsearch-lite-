---
name: knowledge_query
description: 查询本地 FAISS 知识库中的历史搜索结果，不联网
tools:
  - knowledge_query
---

# Knowledge Query — 本地知识库查询

## 什么时候用

- 用户问的内容**之前搜过**，想快速回顾
- 不需要最新信息，历史数据就够用
- 想要**零延迟**、不消耗搜索 API 的回答
- 检查知识库里有没有某个话题的积累

## 什么时候不用

- 需要最新信息 → 用 `deep_search` 或 `quick_search`
- 知识库还是空的（首次使用）→ 先用 `deep_search` 积累数据

## 工作原理

直接用问题的 embedding 在 FAISS 向量库中做相似度检索，返回最相关的 k 条历史搜索摘要。

- 不调用 LLM（除了 embedding）
- 不联网
- 毫秒级响应

## 示例

```
输入: "AI 政策"
输出: 之前搜索 "中美AI政策对比" 时存入的相关摘要

输入: "量子计算"
输出: "本地知识库暂无相关数据，请先使用 deep_search 搜索"
```
