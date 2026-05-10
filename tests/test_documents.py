"""文档管线：智能分块 + Milvus 多集合 + 入库检索"""
import sys
sys.path.insert(0, ".")

# 1. 测试智能分块
print("=== 测试智能分块 ===")
from core.chunker import chunk_text, chunk_document_chunks

text = """# 人工智能搜索引擎

AI搜索引擎是一种基于大语言模型的信息检索系统。它能够理解用户的自然语言查询，将复杂问题分解为多个子问题，并行搜索多个来源，最终综合生成带引用的答案。

## 核心优势

与传统搜索引擎相比，AI搜索引擎的优势在于：
1. 理解意图而非关键词匹配
2. 自动综合多个信息源
3. 生成结构化的答案而非链接列表

## 技术架构

典型的AI搜索引擎采用多Agent架构，包括规划器（Planner）、搜索器（Searcher）、评估器（Evaluator）等组件。这些组件通过状态图（StateGraph）进行协调，实现复杂的搜索推理流程。"""

chunks = chunk_text(text, source_file="test.md")
print(f"输入文本: {len(text)} 字符")
print(f"分块数量: {len(chunks)}")
for c in chunks:
    print(f"  [{c.index}] {len(c.text)} chars: {c.text[:60]}...")

# 2. 测试 Milvus 多集合
print("\n=== 测试 Milvus 多集合 ===")
from core.vector_store import get_vector_store

store = get_vector_store()
print("VectorStore 初始化成功，集合已创建")

# 查看集合状态
for name in ["document_chunks", "search_cache", "user_memory"]:
    stats = store.client.get_collection_stats(name)
    print(f"  {name}: {stats['row_count']} rows")

# 3. 测试文档存入
print("\n=== 测试文档存入 ===")
store.add_document_chunks(chunks, user_id="test_user")
stats = store.client.get_collection_stats("document_chunks")
print(f"存入后 document_chunks: {stats['row_count']} rows")

# 4. 测试检索
print("\n=== 测试检索 ===")
results = store.search_documents("AI搜索引擎有什么优势", k=3, user_id="test_user")
print(f"检索到 {len(results)} 条结果:")
for doc in results:
    score = doc.metadata.get("score", 0)
    print(f"  [{score:.3f}] {doc.page_content[:80]}...")

# 5. 测试搜索缓存
print("\n=== 测试搜索缓存 ===")
store.cache_search_result(
    query="什么是多Agent架构",
    text="多Agent架构是一种将复杂任务分解给多个专业化AI代理协同完成的系统设计模式。",
    source="https://example.com/multi-agent",
)
cache_hits = store.search_cache("多Agent系统怎么工作", k=1, threshold=0.5)
print(f"缓存命中: {len(cache_hits)} 条")
for doc in cache_hits:
    print(f"  [{doc.metadata.get('score', 0):.3f}] {doc.page_content[:80]}...")

print("\n=== 文档管线测试通过 ===")
