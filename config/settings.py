import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM 主模型 ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# ── 评估模型（Evaluator / Reflector 交叉检查）──
EVAL_API_KEY = os.getenv("EVAL_API_KEY", "")
EVAL_BASE_URL = os.getenv("EVAL_BASE_URL", "")
EVAL_MODEL_NAME = os.getenv("EVAL_MODEL_NAME", "")

# ── Embedding ──
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

# ── Milvus ──
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")

# ── MySQL（Phase 7 启用）──
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "mindsearch")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mindsearch_pass")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "mindsearch")

# ── Redis ──
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# ── FastAPI ──
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))
