import logging
from sqlalchemy.orm import Session
from app.core.rag_chain import generate_answer
from app.core.retriever import retrieve
from app.stores import semantic_cache

logger = logging.getLogger(__name__)


def answer_query( db: Session, query: str, document_id: str | None = None, top_k: int = 5,) -> dict:
   
    cached_answer = semantic_cache.check_cache(query, document_id=document_id)
    
    if cached_answer is not None:
        return {
            "answer": cached_answer,
            "cache_hit": True,
            "retrieved_chunks": [],
        }

    retrieved_chunks = retrieve(db, query=query, top_k=top_k, document_id=document_id)
    answer = generate_answer(query, retrieved_chunks)

    semantic_cache.store_answer(query, answer, document_id=document_id)

    return {
        "answer": answer,
        "cache_hit": False,
        "retrieved_chunks": retrieved_chunks,
    }
