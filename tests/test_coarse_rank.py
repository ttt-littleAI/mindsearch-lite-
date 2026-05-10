"""粗排测试 — BM25 + RRF 融合"""
import sys
import importlib.util
spec = importlib.util.spec_from_file_location("coarse_ranker", "core/coarse_ranker.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["coarse_ranker"] = mod
spec.loader.exec_module(mod)
coarse_rank = mod.coarse_rank
_bm25_search = mod._bm25_search

query = "Transformer的自注意力机制是如何工作的"

# 模拟向量召回结果（按向量相似度排序）
vector_results = [
    {"text": "BERT是基于Transformer编码器的预训练语言模型", "score": 0.92},
    {"text": "GPT系列模型使用Transformer解码器架构进行自回归文本生成", "score": 0.90},
    {"text": "卷积神经网络CNN在图像识别领域取得了巨大成功", "score": 0.85},
    {"text": "Transformer模型使用自注意力机制来捕获序列中任意两个位置之间的依赖关系，通过Query、Key、Value三个矩阵计算注意力权重", "score": 0.83},
    {"text": "自注意力(Self-Attention)允许输入序列中的每个元素关注其他所有元素，计算公式为Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V", "score": 0.80},
    {"text": "循环神经网络RNN通过隐藏状态传递信息，但存在长距离依赖问题", "score": 0.75},
    {"text": "多头注意力将Q、K、V投影到多个子空间分别计算注意力，再拼接结果，增强模型的表达能力", "score": 0.72},
    {"text": "Python是一种流行的编程语言，广泛用于数据科学和机器学习", "score": 0.60},
]

# BM25 语料 = 同一批文档
bm25_corpus = [{"text": item["text"]} for item in vector_results]

print("=" * 60)
print("测试1: BM25 关键词搜索")
print("=" * 60)
bm25_results = _bm25_search(query, bm25_corpus, top_n=5)
for i, r in enumerate(bm25_results):
    print(f"  [{i+1}] bm25={r['bm25_score']:.4f} | {r['text'][:60]}...")

print()
print("=" * 60)
print("测试2: RRF 融合粗排")
print("=" * 60)
results = coarse_rank(query, vector_results, bm25_corpus=bm25_corpus, top_n=5)
for i, r in enumerate(results):
    print(f"  [{i+1}] rrf={r.score:.4f} | 来源={r.sources} | {r.text[:60]}...")

print()
print("=" * 60)
print("测试3: 纯向量粗排（无BM25）")
print("=" * 60)
results_no_bm25 = coarse_rank(query, vector_results, bm25_corpus=None, top_n=5)
for i, r in enumerate(results_no_bm25):
    print(f"  [{i+1}] rrf={r.score:.4f} | {r.text[:60]}...")

print()
print("--- 验证 ---")
print("预期: 融合后，含'自注意力'关键词的文档排名应高于纯向量召回")
print("  向量排名: BERT(1) > GPT(2) > CNN(3) > 自注意力机制(4) > 自注意力公式(5)")
print("  融合后应: 自注意力相关文档提前，CNN/Python 等无关文档下降")
