import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.models.db import get_db
from app.models.schemas import DocumentListItem, DocumentListResponse, DocumentStatus, DocumentStatusResponse
from app.stores import doc_store, vector_store

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)) -> DocumentListResponse:
    
    documents = doc_store.list_documents(db)
    
    return DocumentListResponse(
        documents=[
            DocumentListItem(
                document_id=doc.id,
                filename=doc.filename,
                status=DocumentStatus(doc.status.value),
                created_at=doc.created_at,
            )
            for doc in documents
        ]
    )


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str, db: Session = Depends(get_db)) -> DocumentStatusResponse:
    
    document = doc_store.get_document(db, document_id)
    
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    chunk_count = doc_store.count_chunks_for_document(db, document_id)

    return DocumentStatusResponse(
        document_id=document.id,
        filename=document.filename,
        status=DocumentStatus(document.status.value),
        error_message=document.error_message,
        chunk_count=chunk_count,
        created_at=document.created_at,
        updated_at=document.updated_at,
    )


@router.delete("/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)) -> dict:
    
    document = doc_store.get_document(db, document_id)
    
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        vector_store.delete_vectors_for_document(document_id)

    except Exception:
        logger.exception("Failed to delete Pinecone vectors for document %s", document_id)
        raise HTTPException(
            status_code=502, detail="Failed to delete vectors from Pinecone. Try again."
        )

    doc_store.delete_document(db, document_id)
    
    return {"status": "deleted", "document_id": document_id}
