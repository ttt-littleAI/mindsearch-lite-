"""多模态文档解析 — 基于 MinerU 的统一解析入口

支持: PDF / Word / PPT / Excel / 图片
流程: 文件 → MinerU 本地服务解析 → Markdown + 结构化元素
图片: 走 LLM Vision 描述
"""

import asyncio
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from core.exceptions import UnsupportedFileType

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".bmp", ".webp"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class DocumentChunk:
    content: str
    chunk_index: int
    source_file: str
    file_type: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


def parse_document(file_path: str) -> list[DocumentChunk]:
    """统一文档解析入口（同步）"""
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(ext)

    if ext in IMAGE_EXTENSIONS:
        return _parse_image(path)

    return asyncio.run(_parse_with_mineru(path))


async def _parse_with_mineru(file_path: Path) -> list[DocumentChunk]:
    """用 MinerU 本地服务解析文档（PDF/Word/PPT/Excel）"""
    from mineru.cli.api_client import (
        LocalAPIServer,
        UploadAsset,
        build_parse_request_form_data,
        submit_parse_task,
        wait_for_task_result,
        wait_for_local_api_ready,
        download_result_zip,
    )

    ext = file_path.suffix.lower()
    file_name = file_path.name

    form_data = build_parse_request_form_data(
        lang_list=["ch", "en"],
        backend="pipeline",
        parse_method="auto",
        formula_enable=False,
        table_enable=True,
        server_url=None,
        start_page_id=0,
        end_page_id=None,
        return_md=True,
        return_middle_json=True,
        return_content_list=True,
        return_model_output=False,
        return_images=True,
        response_format_zip=True,
        return_original_file=False,
    )

    assets = [UploadAsset(path=file_path, upload_name=file_name)]

    server = LocalAPIServer()
    try:
        base_url = server.start()

        import httpx
        async with httpx.AsyncClient() as client:
            await wait_for_local_api_ready(client, server)
            submit_resp = await submit_parse_task(base_url, assets, form_data)

            await wait_for_task_result(
                client=client,
                submit_response=submit_resp,
                task_label=file_name,
                timeout_seconds=300.0,
            )

            zip_path = await download_result_zip(
                client=client,
                submit_response=submit_resp,
                task_label=file_name,
            )

            return _extract_chunks_from_zip(zip_path, file_name, ext)
    finally:
        server.stop()


def _extract_chunks_from_zip(zip_path: Path, file_name: str, ext: str) -> list[DocumentChunk]:
    """从 MinerU 输出的 ZIP 中提取结构化内容"""
    import zipfile

    chunks = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        md_files = [f for f in zf.namelist() if f.endswith(".md")]
        json_files = [f for f in zf.namelist() if f.endswith("content_list.json")]

        if json_files:
            with zf.open(json_files[0]) as f:
                content_list = json.loads(f.read().decode("utf-8"))

            for idx, item in enumerate(content_list):
                content = item.get("text", "") or item.get("content", "")
                if not content.strip():
                    continue

                chunks.append(DocumentChunk(
                    content=content.strip(),
                    chunk_index=idx,
                    source_file=file_name,
                    file_type=ext.lstrip("."),
                    page_number=item.get("page_idx"),
                    metadata={
                        "type": item.get("type", "text"),
                    },
                ))

        elif md_files:
            with zf.open(md_files[0]) as f:
                md_content = f.read().decode("utf-8")

            sections = _split_markdown_sections(md_content)
            for idx, section in enumerate(sections):
                if not section.strip():
                    continue
                chunks.append(DocumentChunk(
                    content=section.strip(),
                    chunk_index=idx,
                    source_file=file_name,
                    file_type=ext.lstrip("."),
                    page_number=None,
                    metadata={"type": "markdown_section"},
                ))

    if not chunks:
        chunks.append(DocumentChunk(
            content=f"[文档 {file_name} 解析结果为空]",
            chunk_index=0,
            source_file=file_name,
            file_type=ext.lstrip("."),
        ))

    return chunks


def _split_markdown_sections(md: str) -> list[str]:
    """按标题拆分 markdown 为段落"""
    import re
    sections = re.split(r"\n(?=#{1,3}\s)", md)
    return [s for s in sections if s.strip()]


def _parse_image(file_path: Path) -> list[DocumentChunk]:
    """用 LLM Vision 描述图片内容"""
    from tools.image_parser import describe_image

    description = describe_image(str(file_path))
    return [DocumentChunk(
        content=description,
        chunk_index=0,
        source_file=file_path.name,
        file_type=file_path.suffix.lstrip("."),
        metadata={"type": "image_description"},
    )]
