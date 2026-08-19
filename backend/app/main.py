import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import routes_chat, routes_documents, routes_ingest
from app.config import settings
from app.models.db import init_db
from app.stores.vector_store import ensure_index_exists

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("Starting up: initializing SQLite docstore...")
    init_db()
    logger.info("SQLite docstore ready at %s", settings.sqlite_db_path)

    if settings.pinecone_api_key:
        logger.info("Verifying Pinecone index...")
        try:
            ensure_index_exists()
        except Exception:
            
            logger.exception(
                "Could not verify/create Pinecone index. "
                "Ingestion and chat requests will fail until this is fixed."
            )
    else:
        logger.warning("PINECONE_API_KEY not set — skipping Pinecone startup check.")

    yield

    
    logger.info("Shutting down.")


app = FastAPI(
    title="Multimodal RAG",
    description="FastAPI backend for a multimodal (text/table/image) RAG pipeline over PDFs.",
    version="0.1.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Please try again."},
    )


app.include_router(routes_ingest.router)
app.include_router(routes_chat.router)
app.include_router(routes_documents.router)


@app.get("/", tags=["health"])
def root() -> dict:
    return {"status": "ok", "service": "multimodal-rag-api", "env": settings.app_env}


@app.get("/health", tags=["health"])
def health() -> dict:
    return {"status": "healthy"}
