import logging
from app.core import summarization
from app.core.embeddings import embed_texts
from app.core.partitioning import RawChunkType, partition_pdf_file
from app.models.db import db_session
from app.models.db_models import ChunkType, DocumentStatus
from app.stores import doc_store, vector_store

logger = logging.getLogger(__name__)

def process_document(document_id: str, file_path: str) -> None:
    
    try:
        with db_session() as db:
            doc_store.set_document_status(db, document_id, DocumentStatus.PROCESSING)

        logger.info("Starting ingestion for document %s (%s)", document_id, file_path)
        raw_chunks = partition_pdf_file(file_path)

        
        summaries: list[str] = []
        
        for raw_chunk in raw_chunks:
        
            if raw_chunk.chunk_type == RawChunkType.TEXT:
                summaries.append(summarization.summarize_text(raw_chunk.text_content or ""))
                
            elif raw_chunk.chunk_type == RawChunkType.TABLE:
                summaries.append(summarization.summarize_table(raw_chunk.text_content or ""))
           
            elif raw_chunk.chunk_type == RawChunkType.IMAGE:
                summaries.append(summarization.summarize_image(raw_chunk.image_bytes or b""))

        embeddings = embed_texts(summaries)

       
        vector_items = []
        
        with db_session() as db:
            
            for index, (raw_chunk, summary, embedding) in enumerate( zip(raw_chunks, summaries, embeddings) ):
                
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
