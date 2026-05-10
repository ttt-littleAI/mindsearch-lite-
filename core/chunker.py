"""智能分块器 — 将文档内容切分为语义完整的检索单元

策略:
  1. 先按自然段落 / Markdown标题边界切分（结构感知）
  2. 过短段落向前合并
  3. 过长段落用 embedding 相似度找语义断点（默认）；
     embedding 不可用时回退到滑动窗口硬切

语义切分原理：
  - 把长段落切成句子，对每句算 BGE embedding
  - 计算相邻句子的余弦相似度
  - 在相似度低谷（话题切换处）插入断点
  - 受 max_chars 上限约束
"""

import math
import re
import uuid
from dataclasses import dataclass, field

CHUNK_MAX_CHARS = 800
CHUNK_OVERLAP_CHARS = 100
CHUNK_MIN_CHARS = 50
SEMANTIC_MIN_SENTENCES = 4         # 少于此句数不做语义切分
SEMANTIC_ZSCORE_THRESHOLD = -1.0   # z-score 低于此值视为反常低相似度（话题切换）

HEADING_PATTERN = re.compile(r"\n(?=#{1,3}\s)")
PARAGRAPH_PATTERN = re.compile(r"\n{2,}")
SENTENCE_PATTERN = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z])")

_embedder = None


def _get_embedder():
    """懒加载 BGE embedder（首次调用时初始化）。失败返回 None 触发回退。"""
    global _embedder
    if _embedder is False:
        return None
    if _embedder is not None:
        return _embedder
    try:
        from langchain_openai import OpenAIEmbeddings
        from config import EMBEDDING_API_KEY, EMBEDDING_BASE_URL, EMBEDDING_MODEL
        _embedder = OpenAIEmbeddings(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_BASE_URL,
            model=EMBEDDING_MODEL,
        )
        return _embedder
    except Exception:
        _embedder = False
        return None


@dataclass
class Chunk:
    text: str
    index: int
    source_file: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


@dataclass
class ParentChunkData:
    """父块：粗粒度大块，存全文供 LLM 阅读"""
    parent_id: str
    text: str
    source_file: str
    page_number: int | None = None
    file_type: str = ""


# 父子文档切分配置
PARENT_MAX_CHARS = 1500     # 父块上限（喂 LLM 时上下文足够）
CHILD_MAX_CHARS = 300       # 子块上限（向量检索精度高）


def chunk_text(
    text: str,
    source_file: str = "",
    page_number: int | None = None,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
    min_chars: int = CHUNK_MIN_CHARS,
    use_semantic: bool = True,
) -> list[Chunk]:
    if not text.strip():
        return []

    raw_sections = _split_by_structure(text)
    merged = _merge_short(raw_sections, min_chars)
    if use_semantic and _get_embedder() is not None:
        final_parts = _split_long_semantic(merged, max_chars)
    else:
        final_parts = _split_long(merged, max_chars, overlap)

    return [
        Chunk(
            text=part.strip(),
            index=i,
            source_file=source_file,
            page_number=page_number,
            metadata={},
        )
        for i, part in enumerate(final_parts)
        if part.strip()
    ]


def chunk_text_hierarchical(
    text: str,
    source_file: str = "",
    page_number: int | None = None,
    file_type: str = "",
    parent_max_chars: int = PARENT_MAX_CHARS,
    child_max_chars: int = CHILD_MAX_CHARS,
    use_semantic: bool = True,
) -> tuple[list[ParentChunkData], list[Chunk]]:
    """父子切分：返回 (parents, children)，children 通过 metadata['parent_id'] 关联到 parents。

    流程:
      1. 第一次切分（结构感知 + max_chars=parent_max_chars）→ 父块（~1500字大块）
      2. 对每个父块二次切分（结构感知 + 语义梯度法 + max_chars=child_max_chars）→ 子块（~300字小块）
      3. 子块继承父块的 source_file/page_number，metadata 加入 parent_id

    入库时：父块全文存 MySQL，子块向量化存 Milvus（带 parent_id 字段）
    检索时：用 query 搜子块召回 → 去重 parent_id → 回查父块全文喂 LLM
    """
    if not text.strip():
        return [], []

    parent_texts = chunk_text(
        text,
        source_file=source_file,
        page_number=page_number,
        max_chars=parent_max_chars,
        use_semantic=False,  # 父块只需结构感知，不必算 embedding（省 API 调用）
    )

    parents: list[ParentChunkData] = []
    children: list[Chunk] = []
    child_global_idx = 0

    for p in parent_texts:
        pid = uuid.uuid4().hex
        parents.append(ParentChunkData(
            parent_id=pid,
            text=p.text,
            source_file=source_file,
            page_number=page_number,
            file_type=file_type,
        ))
        # 父块再切子块（这里用语义切分提高检索精度）
        sub_chunks = chunk_text(
            p.text,
            source_file=source_file,
            page_number=page_number,
            max_chars=child_max_chars,
            use_semantic=use_semantic,
        )
        for sc in sub_chunks:
            sc.index = child_global_idx
            sc.metadata["parent_id"] = pid
            sc.metadata["file_type"] = file_type
            children.append(sc)
            child_global_idx += 1

    return parents, children


def chunk_document_chunks(doc_chunks: list) -> list[Chunk]:
    """将 DocumentChunk 列表转换为检索用 Chunk 列表（扁平模式，无父块）"""
    all_chunks: list[Chunk] = []
    idx = 0
    for dc in doc_chunks:
        parts = chunk_text(
            dc.content,
            source_file=dc.source_file,
            page_number=dc.page_number,
        )
        for part in parts:
            part.index = idx
            part.metadata.update(dc.metadata)
            idx += 1
        all_chunks.extend(parts)
    return all_chunks


def chunk_document_chunks_hierarchical(
    doc_chunks: list,
) -> tuple[list[ParentChunkData], list[Chunk]]:
    """将 DocumentChunk 列表转换为父子结构的 (parents, children)。

    每个 DocumentChunk 独立做父子切分，子块继承 DocumentChunk 的 metadata（含 file_type 等）。
    """
    all_parents: list[ParentChunkData] = []
    all_children: list[Chunk] = []
    child_idx = 0
    for dc in doc_chunks:
        parents, children = chunk_text_hierarchical(
            dc.content,
            source_file=dc.source_file,
            page_number=dc.page_number,
            file_type=dc.metadata.get("file_type", ""),
        )
        all_parents.extend(parents)
        for c in children:
            c.index = child_idx
            # 合并上层 metadata（DocumentChunk 自带的元数据）但保留 parent_id
            extra = {k: v for k, v in dc.metadata.items() if k != "parent_id"}
            extra.update(c.metadata)
            c.metadata = extra
            child_idx += 1
        all_children.extend(children)
    return all_parents, all_children


def _split_by_structure(text: str) -> list[str]:
    if re.search(r"^#{1,3}\s", text, re.MULTILINE):
        sections = HEADING_PATTERN.split(text)
    else:
        sections = PARAGRAPH_PATTERN.split(text)
    return [s for s in sections if s.strip()]


def _merge_short(sections: list[str], min_chars: int) -> list[str]:
    if not sections:
        return []
    merged = [sections[0]]
    for s in sections[1:]:
        if len(merged[-1]) < min_chars:
            merged[-1] = merged[-1] + "\n\n" + s
        else:
            merged.append(s)
    return merged


def _split_long(sections: list[str], max_chars: int, overlap: int) -> list[str]:
    result = []
    for section in sections:
        if len(section) <= max_chars:
            result.append(section)
            continue
        start = 0
        while start < len(section):
            end = start + max_chars
            if end < len(section):
                break_point = section.rfind("\n", start, end)
                if break_point <= start:
                    break_point = section.rfind("。", start, end)
                if break_point <= start:
                    break_point = section.rfind(". ", start, end)
                if break_point > start:
                    end = break_point + 1
            result.append(section[start:end])
            start = end - overlap if end < len(section) else end
    return result


# ── 语义切分 ────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """按中英文句末标点切句，丢弃空句"""
    sents = [s.strip() for s in SENTENCE_PATTERN.split(text) if s and s.strip()]
    return sents


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _find_semantic_breaks(sims: list[float]) -> set[int]:
    """Z-score 自适应法找语义断点：返回需要在「该位置之后」切分的索引集合。

    判定：相似度的 z-score 低于 SEMANTIC_ZSCORE_THRESHOLD（默认 -1.0）即视为
    "相对于本文档基线的反常低相似度"，对应话题切换。

    与百分位法的关键区别：
      - 百分位法强制切固定比例（每文档都切 25%）
      - Z-score 法只在真有"显著低于均值"的位置切（话题平稳的长文可能 0 个断点；
        话题密集切换的文档可能多个断点）
      - 自适应文本特性，不依赖 BGE 模型的绝对相似度 baseline
    """
    if len(sims) < 2:
        return set()
    n = len(sims)
    mean = sum(sims) / n
    var = sum((s - mean) ** 2 for s in sims) / n
    std = math.sqrt(var)
    if std < 1e-6:
        return set()  # 文本相似度高度均匀，没有可识别的话题切换
    return {i for i, s in enumerate(sims) if (s - mean) / std < SEMANTIC_ZSCORE_THRESHOLD}


def _split_long_semantic(sections: list[str], max_chars: int) -> list[str]:
    """对过长段落用 embedding 梯度法找语义断点；过短段落直接放过。

    回退策略：embedder 调用失败 → 该段落改用滑窗硬切。
    """
    embedder = _get_embedder()
    result = []
    for section in sections:
        if len(section) <= max_chars:
            result.append(section)
            continue

        sentences = _split_into_sentences(section)
        if len(sentences) < SEMANTIC_MIN_SENTENCES:
            result.extend(_split_long([section], max_chars, CHUNK_OVERLAP_CHARS))
            continue

        try:
            vectors = embedder.embed_documents(sentences)
        except Exception:
            result.extend(_split_long([section], max_chars, CHUNK_OVERLAP_CHARS))
            continue

        sims = [_cosine(vectors[i], vectors[i + 1]) for i in range(len(vectors) - 1)]
        break_after = _find_semantic_breaks(sims)

        # 按断点 + max_chars 上限组装 chunks
        current = ""
        for i, sent in enumerate(sentences):
            tentative = current + sent if not current else current + " " + sent
            if len(tentative) > max_chars and current:
                result.append(current)
                current = sent
            else:
                current = tentative
                if i in break_after and len(current) >= CHUNK_MIN_CHARS:
                    result.append(current)
                    current = ""
        if current:
            result.append(current)
    return result
