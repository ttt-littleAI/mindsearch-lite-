"""MindSearch v2 — FastAPI 入口"""

import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 统一 logging 配置：项目内所有 logger.warning/info 输出到 stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import health, search, documents, memory, users, cache, skills, stream, tools

app = FastAPI(
    title="MindSearch v2",
    description="通用 AI 搜索平台 — 搜一切",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(search.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(memory.router, prefix="/api")
app.include_router(users.router, prefix="/api")
app.include_router(cache.router, prefix="/api")
app.include_router(skills.router, prefix="/api")
app.include_router(stream.router, prefix="/api")
app.include_router(tools.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
