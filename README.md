# MindSearch-Lite

<!-- 推送到 GitHub 后把 USERNAME/REPO 替换为你的真实仓库路径 -->
[![tests](https://github.com/USERNAME/REPO/actions/workflows/test.yml/badge.svg)](https://github.com/USERNAME/REPO/actions/workflows/test.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于 LangGraph 的多 Agent 智能搜索引擎，实现问题拆解、并行搜索、RAG 增强、反思优化的完整搜索管线。

## 架构概览

```
用户查询
  │
  ▼
┌─────────────────────────────────────────────┐
│  Layer 1: Input Understanding               │
│  意图识别 + 时效性分类 + 技能匹配           │
│  置信度 < 0.6 → 主动引导用户澄清           │
└──────────────────┬──────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   REALTIME     STABLE    PERSONAL
   纯网络搜索   混合搜索   本地文档
        │          │          │
        ▼          ▼          ▼
┌─────────────────────────────────────────────┐
│  Layer 2: Multi-Agent Search (LangGraph)    │
│                                             │
│  Planner → RAG Retriever → Searcher         │
│     ↑         → Citation Mapper             │
│     └── Evaluator (不够→重新规划)            │
│              → Synthesizer → Reflector      │
│                    ↑           │             │
│                    └─ revise ──┘             │
└──────────────────┬──────────────────────────┘
                   │
┌─────────────────────────────────────────────┐
│  Layer 3: Hierarchical Retrieval + Ranking  │
│  搜子块（小，向量精准命中）                 │
│  → 去重 parent_id → 回查父块（大，喂 LLM）  │
│  → BM25 + RRF 粗排 → bge-reranker 精排     │
└─────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  Layer 4: RAGAS 评估实时入库                │
│  faithfulness / answer_relevancy /          │
│  context_precision / context_recall         │
│  → MySQL search_logs（可 SQL 分析）         │
└─────────────────────────────────────────────┘
```

## 核心特性

- **LangGraph 多 Agent 协作** — Planner/Searcher/Evaluator/Reflector/Synthesizer 等 8 节点状态图，支持反思循环和 DAG 拓扑规划
- **混合路由：规则前置 + LLM 兜底** — 高置信度信号词（"今天/我的文档"等）正则短路，跳过 LLM 调用降时延降成本；其余走 LLM 时效性分类（REALTIME/STABLE/PERSONAL，60 条测试集 98.3% 准确率）
- **Skill 配置驱动整条管线** — 5 个内置技能（学术搜索/新闻简报/对比分析/文档问答/摘要总结），命中即决定路由策略 + Synthesizer 输出格式 + 子问题数量上限 + 专属 system_prompt（不只是 print 一个名字）
- **父子文档检索** — 入库时按结构切大块（父，~1500 字存 MySQL）+ z-score 自适应语义切小块（子，~300 字向量入 Milvus）；检索时搜子块去重 parent_id 回查父块全文喂 LLM，兼顾召回精度和上下文完整性
- **三级排序管线** — 向量召回 Top30 + BM25(jieba) Top30 → RRF 粗排 Top15 → bge-reranker-v2-m3 精排 Top5
- **全局引用映射** — 每个 Searcher 输出局部引用 [n]，Citation Mapper 节点统一映射为全局编号 [N]，Synthesizer 用全局编号生成参考来源
- **RAGAS 实时入库评估** — 每次搜索后调 LLM 算 4 维分数（faithfulness / answer_relevancy / context_precision / context_recall）写入 search_logs，可 SQL 分析回答质量趋势
- **Redis 缓存** — 按路由类型动态 TTL（REALTIME 1h / STABLE 24h），热词统计，滑动窗口限流（30 req/min/user）
- **MCP 协议** — Server 暴露 6 个搜索工具，Client 提供 6 个外部工具（计算器、代码运行、翻译等）
- **SSE 流式输出** — 实时推送搜索进度（技能匹配 → 缓存检查 → 路由 → 搜索 → 完成）
- **四模型分离** — DeepSeek（生成）+ Qwen（评估/反思）+ BGE（Embedding）+ bge-reranker（精排）
- **多格式文档解析** — 基于 MinerU 支持 PDF/Word/PPT/Excel + OCR 图片识别

## 技术栈

| 组件 | 技术 |
|------|------|
| Agent 框架 | LangGraph StateGraph |
| 生成模型 | DeepSeek (兼容 OpenAI API) |
| 评估模型 | Qwen (交叉验证) |
| Embedding | BGE-large-zh-v1.5 (SiliconFlow) |
| 精排模型 | bge-reranker-v2-m3 (SiliconFlow) |
| 向量数据库 | Milvus 2.5 |
| 缓存 | Redis 7 |
| 关系数据库 | MySQL 8.0 |
| Web 框架 | FastAPI + Gradio |
| 网络搜索 | DuckDuckGo |
| 文档解析 | MinerU |
| 中文分词 | jieba |

## 项目结构

```
mindsearch-lite/
├── agents/                 # 多 Agent 实现
│   ├── multi_agent.py      # LangGraph 状态图编排（8 节点）
│   ├── planner.py          # DAG 拓扑规划器
│   ├── searcher.py         # 并行搜索执行器
│   ├── rag.py              # RAG 检索增强
│   └── react_agent.py      # ReAct 推理 Agent
├── api/routes/             # FastAPI 路由
│   ├── search.py           # 搜索接口
│   ├── stream.py           # SSE 流式接口
│   ├── documents.py        # 文档管理
│   ├── cache.py            # 缓存管理
│   ├── skills.py           # 技能查询
│   └── tools.py            # MCP 工具调用
├── core/                   # 核心逻辑
│   ├── search_engine.py    # 统一搜索入口（三层管线）
│   ├── router.py           # 时效性路由器
│   ├── vector_store.py     # Milvus 向量存储
│   ├── coarse_ranker.py    # BM25 + RRF 粗排
│   ├── reranker.py         # LLM 精排
│   ├── cache.py            # Redis 缓存层
│   ├── skills.py           # 技能匹配系统
│   ├── memory.py           # AI 记忆管理
│   ├── database.py         # MySQL ORM
│   └── evaluator.py        # 回答质量评估
├── tools/                  # 工具集
│   ├── search.py           # DuckDuckGo 搜索
│   ├── document_parser.py  # 文档解析（MinerU）
│   ├── image_parser.py     # 图片 OCR
│   └── mcp_tools.py        # MCP Client 工具
├── config/                 # 配置管理
├── tests/                  # 测试套件
├── app.py                  # Gradio UI 入口
├── server.py               # FastAPI 入口
├── mcp_server.py           # MCP Server 入口
├── cli.py                  # 命令行入口
└── docker-compose.yml      # 基础设施编排
```

## 快速开始

### 1. 启动基础设施

```bash
docker-compose up -d
```

启动 Milvus（向量数据库）、Redis（缓存）、MySQL（用户数据）。

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 API Key
```

需要配置：
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` — 主生成模型（DeepSeek / OpenAI 兼容）
- `EMBEDDING_API_KEY` — Embedding 模型（SiliconFlow BGE）
- `EVAL_API_KEY`（可选）— 评估模型（Qwen），不配则用主模型

### 3. 安装依赖

```bash
# 推荐：开发环境（精确版本快照，含传递依赖）
pip install -r requirements.lock.txt

# 或：仅装顶层声明依赖（版本已用 == 锁定）
pip install -r requirements.txt
```

### 4. 启动服务

```bash
# 方式一：Gradio Web UI
python app.py          # http://localhost:7860

# 方式二：FastAPI REST API
python server.py       # http://localhost:8000

# 方式三：命令行
python cli.py

# 方式四：MCP Server（供 Claude Code / Cursor 调用）
python mcp_server.py
```

## API 接口

### 搜索

```
POST /api/search
{
  "question": "什么是Transformer的自注意力机制",
  "search_mode": "auto",   // auto | web | knowledge | hybrid
  "user_id": "user_001"
}
```

### SSE 流式搜索

```
GET /api/search/stream?question=...&user_id=...
```

事件流：`skill` → `cache_hit` → `routing` → `searching` → `done`

### 其他接口

- `POST /api/documents/upload` — 上传文档
- `GET /api/cache/stats` — 缓存统计
- `GET /api/skills` — 技能列表
- `POST /api/tools/call` — 调用 MCP 工具
- `GET /api/health` — 健康检查

## MCP Server

配置 Claude Code 或 Cursor 连接：

```json
{
  "mcpServers": {
    "mindsearch": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/path/to/mindsearch-lite"
    }
  }
}
```

暴露工具：`smart_search` / `web_search` / `knowledge_search` / `document_parse` / `memory_recall` / `hot_queries`

## 评估体系

基于 RAGAS 四维度评估框架，**每次搜索完成后实时入库**（不是离线脚本），可用 SQL 分析回答质量随路由/技能/时间的变化趋势：

| 维度 | 说明 | MySQL 字段 |
|------|------|------|
| Faithfulness | 回答是否忠于检索到的上下文，不编造 | `eval_faithfulness` |
| Answer Relevancy | 回答是否切题（LLM-as-Judge） | `eval_answer_relevancy` |
| Context Precision | 检索结果中相关文档的排序质量 | `eval_context_precision` |
| Context Recall | 标准答案中的关键事实是否被检索覆盖 | `eval_context_recall` |
| Overall | 加权综合分（faith×0.3 + relev×0.3 + prec×0.2 + recall×0.2） | `eval_overall` |

采用交叉模型验证：DeepSeek 生成回答，Qwen 评估质量，避免自评偏差。

**示例 SQL — 查最近 STABLE 路由回答质量趋势：**
```sql
SELECT DATE(created_at) AS day, AVG(eval_overall) AS quality
FROM search_logs WHERE route = 'STABLE' AND eval_overall IS NOT NULL
GROUP BY day ORDER BY day DESC LIMIT 7;
```

## 工程化

- **测试**：`tests/` 按模块组织（test_router.py / test_chunker.py / test_skills.py / test_evaluator.py / test_database.py / test_memory.py / test_documents.py 等），配 `pytest.ini` + `conftest.py`，运行 `pytest tests/`（多数测试需启动 docker-compose 与 .env）
- **日志**：项目内异常路径用 `logging.getLogger(__name__).warning(...)` 上报（不再 `except: pass` 静默吞），server.py 入口统一 `basicConfig`
- **依赖锁定**：`requirements.txt` 用 `==` 锁主依赖；`requirements.lock.txt` 是完整 freeze 快照供严格复现
- **数据库迁移**：首次部署后调 `core.database.init_db()` 自动建表（含 `parent_chunks` 和 `search_logs.eval_*` 字段）；老数据库需手动 `ALTER TABLE` 加列

## License

MIT

