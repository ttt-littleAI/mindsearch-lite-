"""MySQL 多租户：用户 / 搜索日志 / 文档元数据"""
import sys
sys.path.insert(0, ".")

# 1. 建表
print("=== 初始化数据库 ===")
from core.database import init_db, ensure_user, log_search, log_document, get_user_stats, get_search_history
init_db()
print("表创建成功")

# 2. 创建用户
print("\n=== 用户管理 ===")
ensure_user("user_alice")
ensure_user("user_bob")
stats = get_user_stats("user_alice")
print(f"Alice: {stats}")

# 3. 搜索日志
print("\n=== 搜索日志 ===")
log_search("user_alice", "什么是Transformer", "STABLE", "hybrid", "web", 500, 1234.5)
log_search("user_alice", "最新GPT-5消息", "REALTIME", "web_only", "web", 800, 2345.6)
log_search("user_bob", "我的论文里的方法", "PERSONAL", "local_only", "user_docs", 300, 567.8)

stats = get_user_stats("user_alice")
print(f"Alice搜索次数: {stats['search_count']}")

history = get_search_history("user_alice")
print(f"Alice搜索历史: {len(history)} 条")
for h in history:
    print(f"  [{h['route']}/{h['strategy']}] {h['question']} ({h['duration_ms']:.0f}ms)")

# 4. 文档日志
print("\n=== 文档日志 ===")
log_document("user_alice", "report.pdf", "pdf", 1024000, 15)
stats = get_user_stats("user_alice")
print(f"Alice文档数: {stats['doc_count']}")

# 5. 多租户隔离
print("\n=== 多租户隔离 ===")
bob_stats = get_user_stats("user_bob")
print(f"Bob搜索: {bob_stats['search_count']}, 文档: {bob_stats['doc_count']}")
bob_history = get_search_history("user_bob")
print(f"Bob历史: {len(bob_history)} 条")
for h in bob_history:
    print(f"  [{h['route']}] {h['question']}")

print("\n=== Database 测试完成 ===")
