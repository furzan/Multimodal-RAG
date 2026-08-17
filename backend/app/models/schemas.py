from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field



class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ChunkType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"




# ========================================
# used during the ingestion process

class DocumentUploadResponse(BaseModel):

    document_id: str
    filename: str
    status: DocumentStatus
    message: str = "Document received and queued for processing."


class DocumentStatusResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    error_message: str | None = None
    chunk_count: int = 0
    created_at: datetime
    updated_at: datetime


class DocumentListItem(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]



# ========================================





# ========================================
# used for the chat


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User's question about the ingested documents.")
    document_id: str | None = Field(
        default=None,
        description="Optional: restrict retrieval to a single document. If omitted, searches across all ingested documents.",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve.")


class RetrievedChunkPreview(BaseModel):
    chunk_id: str
    chunk_type: ChunkType
    document_id: str
    page_number: int | None = None
    preview: str


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunkPreview] = []
    cache_hit: bool = False



# ========================================