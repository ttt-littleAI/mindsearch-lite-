from pydantic import BaseModel, Field
from enum import Enum


class SearchMode(str, Enum):
    AUTO = "auto"
    WEB = "web"
    KNOWLEDGE = "knowledge"
    HYBRID = "hybrid"


class SearchRequestBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="搜索问题")
    search_mode: SearchMode = Field(default=SearchMode.AUTO, description="搜索模式")
    user_id: str = Field(default="default", description="用户ID")

    model_config = {"json_schema_extra": {"examples": [{"question": "量子计算的基本原理是什么？"}]}}


class Citation(BaseModel):
    """全局引用结构（与 agents/multi_agent.py:_build_global_citations 对齐）"""
    global_index: int = Field(alias="global_index")
    title: str = ""
    url: str = ""
    snippet: str = ""
    sub_question: str = ""

    model_config = {"populate_by_name": True}


class SearchResponseBody(BaseModel):
    answer: str
    citations: list[Citation] = []
    metadata: dict = {}


class HealthResponse(BaseModel):
    status: str
    milvus: str
    version: str
