from pathlib import Path
import uuid
from fastapi import HTTPException, UploadFile

# Define storage directory using Pathlib
UPLOAD_DIR = Path("./uploads")


def ensure_upload_dir_exists() -> None:
    """Creates the upload directory if it doesn't already exist."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def validate_pdf(file: UploadFile) -> None:
    """Validates that the uploaded file is a PDF."""
    is_pdf_ext = file.filename.lower().endswith(".pdf") if file.filename else False
    is_pdf_mime = file.content_type == "application/pdf"

    if not (is_pdf_ext or is_pdf_mime):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF documents are allowed.",
        )


async def save_pdf_to_disk(file: UploadFile) -> Path:
    """Saves the uploaded PDF stream to disk with a unique filename."""
    ensure_upload_dir_exists()
    validate_pdf(file)

    # Preserve original stem while attaching a UUID prefix to avoid collisions
    original_stem = Path(file.filename).stem if file.filename else "document"
    unique_filename = f"{uuid.uuid4().hex}_{original_stem}.pdf"
    destination_path = UPLOAD_DIR / unique_filename

    try:
        # Read stream asynchronously and write to destination
        contents = await file.read()
        with open(destination_path, "wb") as f:
            f.write(contents)
    except Exception as err:
        raise HTTPException(
            status_code=500, detail=f"Could not save PDF file: {str(err)}"
        ) from err
    finally:
        await file.close()

    return destination_path