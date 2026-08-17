from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse



from backend.app.utils.file_utils import save_pdf_to_disk

app = FastAPI(
    title="PDF Upload API",
    description="Minimal FastAPI backend for uploading and storing PDFs",
    version="1.0.0",
)


@app.post("/api/v1/pdfs/upload")
async def upload_pdf_endpoint(file: UploadFile = File(...)):
    """Accepts a single PDF file upload and saves it locally."""
    saved_path = await save_pdf_to_disk(file)

    return JSONResponse(
        status_code=201,
        content={
            "message": "PDF uploaded successfully",
            "filename": saved_path.name,
            "path": str(saved_path),
        },
    )