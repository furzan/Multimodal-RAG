import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from app.core import summarization
from app.core.embeddings import embed_texts
from app.core.partitioning import RawChunkType, partition_pdf_file
from app.models.db import db_session
from app.models.db_models import ChunkType, DocumentStatus
from app.stores import doc_store, vector_store

logger = logging.getLogger(__name__)

# Workers for parallel summarization.
# Ollama (local) is capped at 3 to avoid VRAM pressure.
# Gemini (API) gets its own pool to avoid blocking behind Ollama.
_OLLAMA_WORKERS = 3
_GEMINI_WORKERS = 2


def _summarize_chunk(raw_chunk) -> str:
    """Dispatch a single chunk to the right summarizer."""
    if raw_chunk.chunk_type == RawChunkType.TEXT:
        return summarization.summarize_text(raw_chunk.text_content or "")
    elif raw_chunk.chunk_type == RawChunkType.TABLE:
        return summarization.summarize_table(raw_chunk.text_content or "")
    elif raw_chunk.chunk_type == RawChunkType.IMAGE:
        return summarization.summarize_image(raw_chunk.image_bytes or b"")
    return ""


def _parallel_summarize(raw_chunks: list) -> list[str]:
    """Summarize all chunks concurrently, preserving order."""
    summaries = [""] * len(raw_chunks)

    # Split into Ollama (text/table) vs Gemini (image) tasks
    ollama_tasks = []  # (index, raw_chunk)
    gemini_tasks = []

    for i, rc in enumerate(raw_chunks):
        if rc.chunk_type in (RawChunkType.TEXT, RawChunkType.TABLE):
            ollama_tasks.append((i, rc))
        elif rc.chunk_type == RawChunkType.IMAGE:
            gemini_tasks.append((i, rc))

    def _run_pool(tasks, max_workers, pool_name):
        if not tasks:
            return
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=pool_name) as pool:
            future_to_idx = {
                pool.submit(_summarize_chunk, rc): idx for idx, rc in tasks
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    summaries[idx] = future.result()
                except Exception:
                    logger.exception("Summarization failed for chunk %d", idx)
                    summaries[idx] = ""

    # Run both pools (they don't share resources so can overlap)
    # But we run sequentially to keep things simpler and avoid
    # overloading the machine.
    _run_pool(ollama_tasks, _OLLAMA_WORKERS, "ollama")
    _run_pool(gemini_tasks, _GEMINI_WORKERS, "gemini")

    return summaries


def process_document(document_id: str, file_path: str) -> None:
    
    try:
        with db_session() as db:
            doc_store.set_document_status(db, document_id, DocumentStatus.PROCESSING)

        logger.info("Starting ingestion for document %s (%s)", document_id, file_path)
        raw_chunks = partition_pdf_file(file_path)

        # Summarize all chunks in parallel
        summaries = _parallel_summarize(raw_chunks)

        # Embed all summaries in one batch call
        embeddings = embed_texts(summaries)

        # Store chunks in DB with a single transaction
        vector_items = []
        
        with db_session() as db:
            
            for index, (raw_chunk, summary, embedding) in enumerate(zip(raw_chunks, summaries, embeddings)):
                
                if raw_chunk.chunk_type == RawChunkType.TEXT:
                
                    chunk = doc_store.add_text_chunk(
                        db,
                        document_id=document_id,
                        content_text=raw_chunk.text_content or "",
                        summary=summary,
                        page_number=raw_chunk.page_number,
                        chunk_index=index,
                        chunk_type=ChunkType.TEXT,
                    )
                
                elif raw_chunk.chunk_type == RawChunkType.TABLE:
                
                    chunk = doc_store.add_text_chunk(
                        db,
                        document_id=document_id,
                        content_text=raw_chunk.text_content or "",
                        summary=summary,
                        page_number=raw_chunk.page_number,
                        chunk_index=index,
                        chunk_type=ChunkType.TABLE,
                    )
                
                else:  
                
                    chunk = doc_store.add_image_chunk(
                        db,
                        document_id=document_id,
                        image_bytes=raw_chunk.image_bytes or b"",
                        summary=summary,
                        page_number=raw_chunk.page_number,
                        chunk_index=index,
                    )

                vector_items.append(
                    {
                        "id": chunk.id,
                        "embedding": embedding,
                        "document_id": document_id,
                        "chunk_type": chunk.chunk_type.value,
                        "page_number": chunk.page_number,
                    }
                )

            # Single commit for all chunks (flush happened inside add_*_chunk)
            db.commit()

        vector_store.upsert_summary_vectors(vector_items)

        with db_session() as db:
            doc_store.set_document_status(db, document_id, DocumentStatus.DONE)

        logger.info(
            "Finished ingestion for document %s: %d chunks processed.",
            document_id,
            len(raw_chunks),
        )

    except Exception as exc:
        logger.exception("Ingestion failed for document %s", document_id)
        with db_session() as db:
            doc_store.set_document_status(
                db, document_id, DocumentStatus.FAILED, error_message=str(exc)
            )
