import logging
import uuid
from pathlib import Path
from fastapi import UploadFile
from app.config import UPLOADS_DIR, settings
logger = logging.getLogger(__name__)


class UnsupportedFileTypeError(Exception):
    pass
class FileTooLargeError(Exception):
    pass


def save_upload(file: UploadFile, file_bytes: bytes) -> tuple[str, str]:
   
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError("Only PDF files are supported.")

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise FileTooLargeError(
            f"File exceeds the {settings.max_upload_size_mb}MB upload limit."
        )

    
    safe_filename = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
    dest_path = UPLOADS_DIR / safe_filename
    dest_path.write_bytes(file_bytes)

    logger.info("Saved upload '%s' to %s (%d bytes)", file.filename, dest_path, len(file_bytes))
    
    return str(dest_path), file.filename
