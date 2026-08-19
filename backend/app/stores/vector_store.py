import logging
from pinecone import Pinecone, ServerlessSpec
from app.config import settings
logger = logging.getLogger(__name__)
_pc: Pinecone | None = None


def get_pinecone_client() -> Pinecone:
    
    global _pc
    if _pc is None:
        if not settings.pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Add it to your .env file."
            )
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    
    return _pc


def ensure_index_exists() -> None:
    
  
    pc = get_pinecone_client()
    index_name = settings.pinecone_index_name

    if pc.has_index(index_name):
        logger.info("Pinecone index '%s' already exists.", index_name)
        return

    logger.info(
        "Creating Pinecone index '%s' (dim=%d, cloud=%s, region=%s)...",
        index_name,
        settings.pinecone_embedding_dim,
        settings.pinecone_cloud,
        settings.pinecone_region,
    )
    pc.create_index(
        name=index_name,
        dimension=settings.pinecone_embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )
    logger.info("Pinecone index '%s' created.", index_name)


def get_index():
    global _index
    if _index is None:
        pc = get_pinecone_client()
        _index = pc.Index(settings.pinecone_index_name)
    return _index


_index = None


def upsert_summary_vectors( items: list[dict],) -> None:
    
    """
    Upsert a batch of (chunk_id, embedding, metadata) into Pinecone.

    Each item in `items` must have the shape:
        {
            "id": str,               # chunk.id from SQLite
            "embedding": list[float],
            "document_id": str,
            "chunk_type": str,       # "text" | "table" | "image"
            "page_number": int | None,
        }
    """
    
    if not items:
        return

    index = get_index()
    
    vectors = [
        (
            item["id"],
            item["embedding"],
            {
                "document_id": item["document_id"],
                "chunk_type": item["chunk_type"],
                "page_number": item.get("page_number") or -1,
            },
        )
        for item in items
    ]

  
    batch_size = 100
  
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch)

    logger.info("Upserted %d vectors into Pinecone.", len(vectors))




def query_similar_chunks( query_embedding: list[float], top_k: int = 5, document_id: str | None = None, ) -> list[dict]:
    
    
    index = get_index()

    query_filter = None
    
    if document_id:
        query_filter = {"document_id": {"$eq": document_id}}

    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter=query_filter,
    )

    hits = []
    
    for match in result.matches:
    
        metadata = match.metadata or {}
    
        hits.append(
            {
                "id": match.id,
                "score": match.score,
                "document_id": metadata.get("document_id"),
                "chunk_type": metadata.get("chunk_type"),
                "page_number": metadata.get("page_number"),
            }
        )
    return hits


def delete_vectors_for_document(document_id: str) -> None:
    
    index = get_index()
    index.delete(filter={"document_id": {"$eq": document_id}})
    
    logger.info("Deleted Pinecone vectors for document %s.", document_id)
