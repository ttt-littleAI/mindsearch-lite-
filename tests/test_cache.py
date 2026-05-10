"""Redis 缓存测试"""
import sys
import importlib.util
sys.path.insert(0, ".")

# 绕过 core/__init__.py 的重依赖链
for mod_name, mod_path in [
    ("config.settings", "config/settings.py"),
    ("config", "config/__init__.py"),
    ("core.cache", "core/cache.py"),
]:
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = m
    spec.loader.exec_module(m)

from core.cache import get_search_cache

cache = get_search_cache()

print(f"Redis 可用: {cache.available}")
if not cache.available:
    print("请先启动 Redis: docker-compose up -d redis")
    sys.exit(1)

# 测试写入（不同路由不同 TTL）
test_cases = [
    ("今天的AI新闻有哪些", "REALTIME", "今天发生了很多AI新闻..."),
    ("Transformer的原理是什么", "STABLE", "Transformer使用自注意力机制..."),
    ("我上传的论文讲了什么", "PERSONAL", "根据你的论文..."),
]

print("\n--- 测试写入 ---")
for question, route, answer in test_cases:
    cache.put(
        question=question,
        result={"answer": answer, "citations": [], "metadata": {"route": route}},
        route=route,
    )
    ttl_map = {"REALTIME": "10分钟", "STABLE": "24小时", "PERSONAL": "1小时"}
    print(f"  [{route}] TTL={ttl_map[route]} | {question}")

print("\n--- 测试读取 ---")
for question, route, answer in test_cases:
    result = cache.get(question)
    hit = "命中" if result else "未命中"
    print(f"  {hit} | {question}")
    if result:
        assert result["answer"] == answer, f"内容不匹配: {result['answer']}"

print("\n--- 测试未命中 ---")
result = cache.get("一个从未搜索过的问题")
print(f"  {'命中' if result else '未命中'} | 一个从未搜索过的问题")
assert result is None

print("\n--- 测试失效 ---")
cache.invalidate("今天的AI新闻有哪些")
result = cache.get("今天的AI新闻有哪些")
print(f"  失效后: {'命中' if result else '未命中'} | 今天的AI新闻有哪些")
assert result is None

print("\n--- 热搜排行 ---")
hot = cache.hot_queries(10)
for item in hot:
    print(f"  [{item['count']}次] {item['query']}")

print("\n--- 缓存统计 ---")
stats = cache.stats()
print(f"  可用: {stats['available']}")
print(f"  键数: {stats['keys']}")
print(f"  内存: {stats['memory_used']}")

print("\n所有测试通过!")
