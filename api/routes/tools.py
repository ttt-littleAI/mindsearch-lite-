"""MCP 工具接口 — 查询/调用可用工具"""

from fastapi import APIRouter
from pydantic import BaseModel

from tools.mcp_tools import list_tools_schema, call_tool

router = APIRouter(prefix="/tools", tags=["tools"])


class ToolCallRequest(BaseModel):
    name: str
    arguments: dict = {}


@router.get("")
def get_tools():
    """返回所有可用 MCP 工具的 JSON Schema"""
    return {"tools": list_tools_schema()}


@router.post("/call")
def invoke_tool(body: ToolCallRequest):
    """调用指定工具"""
    result = call_tool(body.name, body.arguments)
    return {"tool": body.name, "result": result}
