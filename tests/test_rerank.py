"""Rerank 精排测试"""
import sys
sys.path.insert(0, ".")
from core.reranker import rerank

query = "Transformer的自注意力机制是如何工作的"

documents = [
    "Python是一种流行的编程语言，广泛用于数据科学和机器学习",
    "Transformer模型使用自注意力机制来捕获序列中任意两个位置之间的依赖关系，通过Query、Key、Value三个矩阵计算注意力权重",
    "卷积神经网络CNN在图像识别领域取得了巨大成功",
    "自注意力(Self-Attention)允许输入序列中的每个元素关注其他所有元素，计算公式为Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V",
    "BERT是基于Transformer编码器的预训练语言模型",
    "循环神经网络RNN通过隐藏状态传递信息，但存在长距离依赖问题",
    "多头注意力将Q、K、V投影到多个子空间分别计算注意力，再拼接结果，增强模型的表达能力",
    "GPT系列模型使用Transformer解码器架构进行自回归文本生成",
]

print(f"Query: {query}\n")
print(f"召回 {len(documents)} 条文档，Rerank 精排取 top5:\n")

results = rerank(query, documents, top_n=5)
for i, r in enumerate(results):
    print(f"  [{i+1}] score={r.score:.4f} | {r.text[:80]}...")

print("\n--- 验证 ---")
print("预期排序: 自注意力公式 > Transformer自注意力 > 多头注意力 > BERT/GPT > 其他")
