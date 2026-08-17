import enum
import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


def _uuid_str() -> str:
    return str(uuid.uuid4())

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)

class Base(DeclarativeBase):
    pass


class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class ChunkType(str, enum.Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


# storest the document that is being injested and also the status 

class Document(Base):

    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=_uuid_str)
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False) 
    status = Column(
        Enum(DocumentStatus), nullable=False, default=DocumentStatus.PENDING
    )
    error_message = Column(Text, nullable=True)  
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    updated_at = Column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )
    chunks = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

# stores the chunks of type text table and image for the image ot only stores the path 

class Chunk(Base):

    __tablename__ = "chunks"

    id = Column(String, primary_key=True, default=_uuid_str)
    document_id = Column(
        String, ForeignKey("documents.id"), nullable=False, index=True
    )
    chunk_type = Column(Enum(ChunkType), nullable=False)
    content_text = Column(Text, nullable=True)
    content_path = Column(String, nullable=True)
    summary = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    chunk_index = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), default=_utcnow)
    document = relationship("Document", back_populates="chunks")
