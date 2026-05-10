"""文档上传与管理 API"""

import logging
import tempfile
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel

from tools.document_parser import parse_document, SUPPORTED_EXTENSIONS
from core.chunker import chunk_document_chunks_hierarchical
from core.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["文档"])


class ChunkResponse(BaseModel):
    content: str
    chunk_index: int
    source_file: str
    page_number: int | None = None


class ParseResponse(BaseModel):
    filename: str
    file_type: str
    chunk_count: int
    stored_in_milvus: bool
    chunks: list[ChunkResponse]


@router.post("/parse", response_model=ParseResponse)
async def upload_and_parse(file: UploadFile = File(...), user_id: str = "default"):
    """上传文件 → 解析 → 智能分块 → 存入 Milvus"""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型: {ext}，支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            doc_chunks = parse_document(str(tmp_path))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"解析失败: {e}")

    parents, chunks = chunk_document_chunks_hierarchical(doc_chunks)

    stored = False
    try:
        store = get_vector_store()
        store.add_document_hierarchical(parents, chunks, user_id=user_id)
        stored = True
    except Exception as e:
        logger.warning("Vector store 入库失败: %s", e)

    try:
        from core.database import log_document
        log_document(
            user_id=user_id,
            filename=file.filename,
            file_type=ext.lstrip("."),
            file_size=0,
            chunk_count=len(chunks),
        )
    except Exception as e:
        logger.warning("MySQL log_document 写入失败: %s", e)

    return ParseResponse(
        filename=file.filename,
        file_type=ext.lstrip("."),
        chunk_count=len(chunks),
        stored_in_milvus=stored,
        chunks=[
            ChunkResponse(
                content=c.text,
                chunk_index=c.index,
                source_file=c.source_file,
                page_number=c.page_number,
            )
            for c in chunks
        ],
    )
