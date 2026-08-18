import logging
from pathlib import Path
from sqlalchemy.orm import Session
from app.config import EXTRACTED_IMAGES_DIR
from app.models.db_models import Chunk, ChunkType, Document, DocumentStatus

logger = logging.getLogger(__name__)


def create_document(db: Session, filename: str, file_path: str) -> Document:
    doc = Document(
        filename=filename,
        file_path=file_path,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def set_document_status(  db: Session, document_id: str, status: DocumentStatus, error_message: str | None = None,) -> Document | None:
    
    doc = db.get(Document, document_id)
    
    if doc is None:
        logger.warning("set_document_status: document %s not found", document_id)
        return None
    
    doc.status = status
    doc.error_message = error_message
    db.commit()
    db.refresh(doc)
    return doc


def get_document(db: Session, document_id: str) -> Document | None:
    
    return db.get(Document, document_id)


def list_documents(db: Session) -> list[Document]:
    
    return db.query(Document).order_by(Document.created_at.desc()).all()


# when a doc is deleted the coresponding chunks are also deleted due to cascading property , image is also deleted via unlink() methord
def delete_document(db: Session, document_id: str) -> bool:
    
    doc = db.get(Document, document_id)
    
    if doc is None:
        return False

    for chunk in doc.chunks:
        if chunk.content_path:
            try:
                Path(chunk.content_path).unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not delete image file %s", chunk.content_path)

    db.delete(doc)
    db.commit()
    return True





def add_text_chunk( db: Session, document_id: str, content_text: str, summary: str, page_number: int | None, chunk_index: int, chunk_type: ChunkType = ChunkType.TEXT, ) -> Chunk:
    
    
    
    chunk = Chunk(
        document_id=document_id,
        chunk_type=chunk_type,
        content_text=content_text,
        summary=summary,
        page_number=page_number,
        chunk_index=chunk_index,
    )
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk


def add_image_chunk(db: Session, document_id: str, image_bytes: bytes, summary: str, page_number: int | None, chunk_index: int, image_ext: str = "png",) -> Chunk:
    
    
    """
    Saves image bytes to disk under EXTRACTED_IMAGES_DIR/<document_id>/
    and stores the resulting path. Image bytes are never stored in SQLite
    directly (see db_models.py docstring for rationale).
    """
    chunk = Chunk(
        document_id=document_id,
        chunk_type=ChunkType.IMAGE,
        summary=summary,
        page_number=page_number,
        chunk_index=chunk_index,
    )
    db.add(chunk)
    db.flush()  # assigns chunk.id without committing yet

    doc_image_dir = EXTRACTED_IMAGES_DIR / document_id
    doc_image_dir.mkdir(parents=True, exist_ok=True)
    image_path = doc_image_dir / f"{chunk.id}.{image_ext}"
    image_path.write_bytes(image_bytes)

    chunk.content_path = str(image_path)
    db.commit()
    db.refresh(chunk)
    return chunk


def get_chunk(db: Session, chunk_id: str) -> Chunk | None:
    
    return db.get(Chunk, chunk_id)


def get_chunks_by_ids(db: Session, chunk_ids: list[str]) -> list[Chunk]:
    
    """
    Batch fetch, preserving no particular order (caller re-sorts to match
    the order chunk_ids came back from Pinecone, since that order encodes
    relevance ranking).
    """
    if not chunk_ids:
        return []
    return db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()


def count_chunks_for_document(db: Session, document_id: str) -> int:
    
    return db.query(Chunk).filter(Chunk.document_id == document_id).count()
