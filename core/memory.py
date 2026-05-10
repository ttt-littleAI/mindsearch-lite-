"""AI 记忆模块 — 短期 / 长期 / 工作记忆

短期记忆: 对话上下文（内存，会话级）
长期记忆: 用户偏好/兴趣（Milvus user_memory 集合，持久化）
工作记忆: 当前搜索任务的中间状态（内存，任务级）
"""

from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME
from core.vector_store import get_vector_store

EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "你是一个用户偏好提取器。从对话中提取用户的持久偏好信息。\n\n"
     "提取以下类型（每行一条，格式 TYPE: content）：\n"
     "- DOMAIN: 用户关注的领域/行业（如：人工智能、金融、医疗）\n"
     "- PREFERENCE: 回答风格偏好（如：喜欢详细解释、偏好中文、要数据支撑）\n"
     "- EXPERTISE: 用户的专业水平（如：深度学习专家、编程初学者）\n"
     "- INTEREST: 具体兴趣点（如：关注Transformer架构、对RAG感兴趣）\n\n"
     "规则：\n"
     "- 只提取明确表现出的偏好，不要猜测\n"
     "- 没有可提取的偏好就输出 NONE\n"
     "- 每条偏好不超过30字"),
    ("human",
     "用户问题: {question}\n"
     "AI回答: {answer}\n\n"
     "请提取用户偏好:"),
])


@dataclass
class WorkingMemory:
    """工作记忆 — 当前搜索任务的中间状态"""
    task_id: str = ""
    original_question: str = ""
    refined_question: str = ""
    search_rounds: int = 0
    key_findings: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)

    def add_finding(self, finding: str):
        if finding not in self.key_findings:
            self.key_findings.append(finding)

    def summary(self) -> str:
        parts = [f"原始问题: {self.original_question}"]
        if self.refined_question != self.original_question:
            parts.append(f"优化问题: {self.refined_question}")
        parts.append(f"搜索轮次: {self.search_rounds}")
        if self.key_findings:
            parts.append("关键发现:\n" + "\n".join(f"  - {f}" for f in self.key_findings[-5:]))
        if self.unresolved:
            parts.append("待解决:\n" + "\n".join(f"  - {u}" for u in self.unresolved))
        return "\n".join(parts)


class AIMemory:
    """AI 记忆管理器"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.store = get_vector_store()
        # 短期记忆
        self.chat_history: list[HumanMessage | AIMessage] = []
        # 工作记忆
        self.working: WorkingMemory = WorkingMemory()

    # ── 短期记忆 ──

    def add_turn(self, question: str, answer: str):
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=answer))

    def get_history(self, max_turns: int = 10) -> list:
        return self.chat_history[-(max_turns * 2):]

    def get_context_summary(self) -> str:
        """将近期对话压缩为摘要"""
        if not self.chat_history:
            return ""
        recent = self.chat_history[-6:]
        parts = []
        for msg in recent:
            role = "用户" if isinstance(msg, HumanMessage) else "AI"
            parts.append(f"{role}: {msg.content[:100]}")
        return "\n".join(parts)

    # ── 长期记忆 ──

    def save_preference(self, text: str, memory_type: str = "INTEREST"):
        self.store.add_user_memory(text, memory_type=memory_type, user_id=self.user_id)

    def recall_preferences(self, query: str, k: int = 5) -> list[dict]:
        docs = self.store.search_user_memory(query, user_id=self.user_id, k=k)
        return [
            {
                "text": doc.page_content,
                "type": doc.metadata.get("memory_type", ""),
                "score": doc.metadata.get("score", 0),
            }
            for doc in docs
        ]

    def extract_and_save_preferences(self, question: str, answer: str):
        """从对话中自动提取用户偏好并存入长期记忆"""
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            temperature=0,
        )
        try:
            raw = (EXTRACT_PROMPT | llm | StrOutputParser()).invoke({
                "question": question,
                "answer": answer[:500],
            })
        except Exception:
            return []

        if "NONE" in raw.upper():
            return []

        saved = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if ":" not in line:
                continue
            parts = line.split(":", 1)
            mem_type = parts[0].strip().upper()
            content = parts[1].strip()
            if mem_type in ("DOMAIN", "PREFERENCE", "EXPERTISE", "INTEREST") and content:
                self.save_preference(content, memory_type=mem_type)
                saved.append({"type": mem_type, "content": content})

        return saved

    # ── 工作记忆 ──

    def start_task(self, question: str):
        import uuid
        self.working = WorkingMemory(
            task_id=str(uuid.uuid4())[:8],
            original_question=question,
            refined_question=question,
        )

    def get_working_context(self) -> str:
        return self.working.summary()

    # ── 综合召回 ──

    def recall_all(self, query: str) -> dict:
        """综合召回：短期上下文 + 长期偏好 + 工作状态"""
        return {
            "chat_context": self.get_context_summary(),
            "preferences": self.recall_preferences(query),
            "working": self.get_working_context() if self.working.task_id else "",
        }


_instances: dict[str, AIMemory] = {}


def get_ai_memory(user_id: str = "default") -> AIMemory:
    if user_id not in _instances:
        _instances[user_id] = AIMemory(user_id=user_id)
    return _instances[user_id]
