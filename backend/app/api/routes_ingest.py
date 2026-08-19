import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.models.db import get_db
from app.models.schemas import DocumentStatus, DocumentUploadResponse
from app.services.ingestion_service import process_document
from app.stores import doc_store
from app.utils.file_utils import FileTooLargeError, UnsupportedFileTypeError, save_upload

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document( background_tasks: BackgroundTasks, file: UploadFile, db: Session = Depends(get_db),) -> DocumentUploadResponse:
    
    file_bytes = await file.read()

    try:
        file_path, original_filename = save_upload(file, file_bytes)
    
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    
    except FileTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc

    document = doc_store.create_document(db, filename=original_filename, file_path=file_path)

    background_tasks.add_task(process_document, document.id, file_path)

    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=DocumentStatus(document.status.value),
    )
