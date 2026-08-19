import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from app.models.db import get_db
from app.models.db_models import ChunkType as DbChunkType
from app.models.schemas import ChatRequest, ChatResponse, ChunkType, RetrievedChunkPreview
from app.services.query_service import answer_query
from app.stores import doc_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


def _preview_for_chunk(retrieved_chunk) -> RetrievedChunkPreview:
    
    if retrieved_chunk.chunk_type == DbChunkType.IMAGE:
        preview_text = "[image]"
    
    else:
        content = retrieved_chunk.content_text or ""
        preview_text = content[:200] + ("..." if len(content) > 200 else "")

    return RetrievedChunkPreview(
        chunk_id=retrieved_chunk.chunk_id,
        chunk_type=ChunkType(retrieved_chunk.chunk_type.value),
        document_id=retrieved_chunk.document_id,
        page_number=retrieved_chunk.page_number,
        preview=preview_text,
    )


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    
    result = answer_query(
        db,
        query=request.query,
        document_id=request.document_id,
        top_k=request.top_k,
    )

    return ChatResponse(
        answer=result["answer"],
        cache_hit=result["cache_hit"],
        retrieved_chunks=[_preview_for_chunk(c) for c in result["retrieved_chunks"]],
    )


@router.get("/image/{chunk_id}")
def get_chunk_image(chunk_id: str, db: Session = Depends(get_db)) -> FileResponse:
    
    chunk = doc_store.get_chunk(db, chunk_id)
    
    if chunk is None or chunk.chunk_type != DbChunkType.IMAGE or not chunk.content_path:
        raise HTTPException(status_code=404, detail="Image not found.")
    
    return FileResponse(chunk.content_path)
