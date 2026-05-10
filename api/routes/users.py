"""用户管理 API"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.database import get_user_stats, get_search_history

router = APIRouter(prefix="/users", tags=["用户"])


class UserStats(BaseModel):
    user_id: str
    search_count: int
    doc_count: int
    created_at: str


class SearchHistoryItem(BaseModel):
    question: str
    route: str
    strategy: str
    duration_ms: float
    created_at: str


@router.get("/{user_id}/stats", response_model=UserStats)
async def user_stats(user_id: str):
    stats = get_user_stats(user_id)
    if not stats.get("exists"):
        raise HTTPException(status_code=404, detail="用户不存在")
    return UserStats(
        user_id=stats["user_id"],
        search_count=stats["search_count"],
        doc_count=stats["doc_count"],
        created_at=stats["created_at"],
    )


@router.get("/{user_id}/history", response_model=list[SearchHistoryItem])
async def search_history(user_id: str, limit: int = 20):
    return [SearchHistoryItem(**item) for item in get_search_history(user_id, limit)]
