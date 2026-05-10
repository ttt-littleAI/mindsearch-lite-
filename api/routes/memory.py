"""用户记忆 API"""

from fastapi import APIRouter
from pydantic import BaseModel

from core.memory import get_ai_memory

router = APIRouter(prefix="/memory", tags=["记忆"])


class PreferenceItem(BaseModel):
    text: str
    type: str
    score: float = 0.0


class MemoryResponse(BaseModel):
    user_id: str
    preferences: list[PreferenceItem]
    chat_turns: int


class SavePreferenceRequest(BaseModel):
    user_id: str = "default"
    text: str
    memory_type: str = "INTEREST"


@router.get("/{user_id}", response_model=MemoryResponse)
async def get_user_memory(user_id: str, query: str = ""):
    """查询用户记忆"""
    ai_mem = get_ai_memory(user_id)
    prefs = ai_mem.recall_preferences(query or "user preferences", k=10)
    return MemoryResponse(
        user_id=user_id,
        preferences=[PreferenceItem(**p) for p in prefs],
        chat_turns=len(ai_mem.chat_history) // 2,
    )


@router.post("/save")
async def save_preference(req: SavePreferenceRequest):
    """手动保存用户偏好"""
    ai_mem = get_ai_memory(req.user_id)
    ai_mem.save_preference(req.text, memory_type=req.memory_type)
    return {"status": "saved", "user_id": req.user_id}
