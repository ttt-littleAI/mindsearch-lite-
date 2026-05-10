"""MySQL 数据库 — 用户/文档/搜索日志/父块

表结构:
  users          — 用户信息 + 配额
  documents      — 上传文档元数据
  search_logs    — 搜索记录（路由决策、耗时、策略）
  parent_chunks  — 父子文档切分中的"父块"全文存储
                   （子块在 Milvus 做向量检索，命中后用 parent_id 回查父块全文喂 LLM）
"""

from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from config import MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE

DATABASE_URL = (
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}"
    f"@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}"
    f"?charset=utf8mb4"
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ── 表定义 ──

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), unique=True, nullable=False, index=True)
    username = Column(String(64), default="")
    email = Column(String(256), default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    search_count = Column(Integer, default=0)
    doc_count = Column(Integer, default=0)

    documents = relationship("DocumentRecord", back_populates="user")
    search_logs = relationship("SearchLog", back_populates="user")


class DocumentRecord(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.user_id"), nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    file_type = Column(String(32), default="")
    file_size = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(32), default="parsed")

    user = relationship("User", back_populates="documents")


class ParentChunk(Base):
    """父块全文存储 — 与 Milvus document_chunks(子块) 通过 parent_id 关联"""
    __tablename__ = "parent_chunks"

    parent_id = Column(String(64), primary_key=True)  # UUID 字符串
    user_id = Column(String(128), ForeignKey("users.user_id"), nullable=False, index=True)
    source_file = Column(String(512), default="")
    file_type = Column(String(32), default="")
    page_number = Column(Integer, default=-1)
    text = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class SearchLog(Base):
    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(128), ForeignKey("users.user_id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    route = Column(String(32), default="")
    strategy = Column(String(32), default="")
    source = Column(String(32), default="")
    answer_length = Column(Integer, default=0)
    duration_ms = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    # RAGAS 四维评估分数（0-1，可选；None 表示未评估）
    eval_faithfulness = Column(Float, nullable=True)
    eval_answer_relevancy = Column(Float, nullable=True)
    eval_context_precision = Column(Float, nullable=True)
    eval_context_recall = Column(Float, nullable=True)
    eval_overall = Column(Float, nullable=True)

    user = relationship("User", back_populates="search_logs")


# ── 初始化 ──

def init_db():
    Base.metadata.create_all(engine)


def get_session():
    return SessionLocal()


# ── 操作函数 ──

def ensure_user(user_id: str) -> User:
    with SessionLocal() as session:
        user = session.query(User).filter_by(user_id=user_id).first()
        if not user:
            user = User(user_id=user_id)
            session.add(user)
            session.commit()
            session.refresh(user)
        return user


def log_search(user_id: str, question: str, route: str, strategy: str,
               source: str, answer_length: int, duration_ms: float,
               eval_scores: dict | None = None):
    """记录搜索日志。eval_scores 可选，含 faithfulness/answer_relevancy/context_precision/context_recall/overall"""
    with SessionLocal() as session:
        ensure_user(user_id)
        log = SearchLog(
            user_id=user_id,
            question=question,
            route=route,
            strategy=strategy,
            source=source,
            answer_length=answer_length,
            duration_ms=duration_ms,
        )
        if eval_scores:
            log.eval_faithfulness = eval_scores.get("faithfulness")
            log.eval_answer_relevancy = eval_scores.get("answer_relevancy")
            log.eval_context_precision = eval_scores.get("context_precision")
            log.eval_context_recall = eval_scores.get("context_recall")
            log.eval_overall = eval_scores.get("overall")
        session.add(log)
        session.query(User).filter_by(user_id=user_id).update(
            {"search_count": User.search_count + 1}
        )
        session.commit()


def log_document(user_id: str, filename: str, file_type: str,
                 file_size: int, chunk_count: int):
    with SessionLocal() as session:
        ensure_user(user_id)
        doc = DocumentRecord(
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            chunk_count=chunk_count,
        )
        session.add(doc)
        session.query(User).filter_by(user_id=user_id).update(
            {"doc_count": User.doc_count + 1}
        )
        session.commit()


def get_user_stats(user_id: str) -> dict:
    with SessionLocal() as session:
        user = session.query(User).filter_by(user_id=user_id).first()
        if not user:
            return {"exists": False}
        return {
            "exists": True,
            "user_id": user.user_id,
            "search_count": user.search_count,
            "doc_count": user.doc_count,
            "created_at": user.created_at.isoformat(),
        }


def save_parent_chunks(parents: list[dict]) -> None:
    """批量保存父块。parents 每条需含: parent_id, user_id, source_file, file_type, page_number, text"""
    if not parents:
        return
    with SessionLocal() as session:
        # 用户必须先存在（外键约束）
        user_ids = {p["user_id"] for p in parents}
        for uid in user_ids:
            ensure_user(uid)
        session.add_all([
            ParentChunk(
                parent_id=p["parent_id"],
                user_id=p["user_id"],
                source_file=p.get("source_file", ""),
                file_type=p.get("file_type", ""),
                page_number=p.get("page_number", -1),
                text=p["text"],
            )
            for p in parents
        ])
        session.commit()


def get_parent_chunks(parent_ids: list[str]) -> dict[str, dict]:
    """按 parent_id 列表批量查父块全文。返回 {parent_id: {text, source_file, ...}}"""
    if not parent_ids:
        return {}
    with SessionLocal() as session:
        rows = session.query(ParentChunk).filter(ParentChunk.parent_id.in_(parent_ids)).all()
        return {
            r.parent_id: {
                "parent_id": r.parent_id,
                "text": r.text,
                "source_file": r.source_file,
                "file_type": r.file_type,
                "page_number": r.page_number,
                "user_id": r.user_id,
            }
            for r in rows
        }


def get_search_history(user_id: str, limit: int = 20) -> list[dict]:
    with SessionLocal() as session:
        logs = (
            session.query(SearchLog)
            .filter_by(user_id=user_id)
            .order_by(SearchLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "question": log.question,
                "route": log.route,
                "strategy": log.strategy,
                "duration_ms": log.duration_ms,
                "created_at": log.created_at.isoformat(),
            }
            for log in logs
        ]
