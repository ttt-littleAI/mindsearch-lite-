from fastapi import APIRouter
from api.schemas import SearchRequestBody, SearchResponseBody, Citation
from core.search_engine import SearchRequest, SearchMode, search_async

router = APIRouter(prefix="/search", tags=["搜索"])


@router.post("", response_model=SearchResponseBody)
async def do_search(body: SearchRequestBody):
    request = SearchRequest(
        question=body.question,
        search_mode=SearchMode(body.search_mode.value),
        user_id=body.user_id,
    )
    result = await search_async(request)
    return SearchResponseBody(
        answer=result.answer,
        citations=[Citation(**c) for c in result.citations],
        metadata=result.metadata,
    )
