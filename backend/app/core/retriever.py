import logging
from dataclasses import dataclass
from sqlalchemy.orm import Session
from app.core.embeddings import embed_query
from app.models.db_models import Chunk, ChunkType
from app.stores import doc_store, vector_store
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    chunk_type: ChunkType
    document_id: str
    page_number: int | None
    score: float
    content_text: str | None
    content_path: str | None


def retrieve(db: Session, query: str, top_k: int = 5, document_id: str | None = None, ) -> list[RetrievedChunk]:
    
    
    query_embedding = embed_query(query)

    hits = vector_store.query_similar_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        document_id=document_id,
    )
    if not hits:
        logger.info("No Pinecone hits for query: %r", query)
        return []

    hit_ids = [hit["id"] for hit in hits]
    chunks_by_id: dict[str, Chunk] = {
        chunk.id: chunk for chunk in doc_store.get_chunks_by_ids(db, hit_ids)
    }
    

    results: list[RetrievedChunk] = []
    
    for hit in hits:
    
        chunk = chunks_by_id.get(hit["id"])
        if chunk is None:
            
            logger.warning(
                "Pinecone hit %s has no matching SQLite chunk; skipping.", hit["id"]
            )
            continue

        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                chunk_type=chunk.chunk_type,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                score=hit["score"],
                content_text=chunk.content_text,
                content_path=chunk.content_path,
            )
        )

    return results
