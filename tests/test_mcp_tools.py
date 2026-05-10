"""MCP 工具测试"""
import sys
import importlib.util

sys.path.insert(0, ".")

for mod_name, mod_path in [
    ("config.settings", "config/settings.py"),
    ("config", "config/__init__.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)

from tools.mcp_tools import list_tools_schema, call_tool

print("=== 已注册 MCP 工具 ===")
for t in list_tools_schema():
    f = t["function"]
    params = list(f["parameters"].get("properties", {}).keys())
    print(f"  [{f['name']}] {f['description'][:40]}... | 参数: {params}")

print("\n=== 测试: calculator ===")
print(call_tool("calculator", {"expression": "sqrt(144) + 2**10"}))

print("\n=== 测试: datetime_now ===")
print(call_tool("datetime_now", {}))

print("\n=== 测试: code_runner ===")
print(call_tool("code_runner", {"code": "print(sum(range(1, 101)))"}))

print("\n=== 测试: redis_query (stats) ===")
print(call_tool("redis_query", {"action": "stats"}))

print("\n=== 测试: redis_query (rate_check) ===")
print(call_tool("redis_query", {"action": "rate_check", "user_id": "test_user"}))

print("\n=== 测试: redis_query (hot) ===")
print(call_tool("redis_query", {"action": "hot"}))

print("\n所有工具测试完成!")
