from fastapi import APIRouter
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    from pymilvus import MilvusClient

    milvus_status = "unknown"
    try:
        client = MilvusClient(uri="http://localhost:19530")
        client.list_collections()
        milvus_status = "connected"
    except Exception:
        milvus_status = "disconnected"

    return HealthResponse(
        status="ok",
        milvus=milvus_status,
        version="2.0.0",
    )
