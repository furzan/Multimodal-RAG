import logging
from redisvl.extensions.cache.llm import SemanticCache
from app.config import settings
logger = logging.getLogger(__name__)
_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    
    global _cache
    
    if _cache is None:
        
        _cache = SemanticCache(
            name=settings.redis_cache_index_name,
            redis_url=settings.redis_url,
            distance_threshold=settings.semantic_cache_distance_threshold,
            filterable_fields=[{"name": "document_id", "type": "tag"}],
        )
        
        logger.info(
            "Redis semantic cache initialized (index=%s, threshold=%.3f).",
            settings.redis_cache_index_name,
            settings.semantic_cache_distance_threshold,
        )
        
    return _cache


def check_cache(query: str, document_id: str | None = None) -> str | None:
   
    
    cache = get_semantic_cache()

    filter_expression = None
    
    if document_id:
        from redisvl.query.filter import Tag

        filter_expression = Tag("document_id") == document_id

    try:
        results = cache.check(prompt=query, filter_expression=filter_expression, num_results=1)
    except Exception:
        
        logger.exception("Semantic cache check failed; treating as cache miss.")
        return None

    if results:
        logger.info("Semantic cache HIT for query: %r", query)
        return results[0]["response"]

    logger.info("Semantic cache MISS for query: %r", query)
    return None


def store_answer(query: str, answer: str, document_id: str | None = None) -> None:
    
    
    cache = get_semantic_cache()
    
    filters = {"document_id": document_id} if document_id else None

    try:
        cache.store(prompt=query, response=answer, filters=filters)
    except Exception:
        
        logger.exception("Semantic cache store failed; continuing without caching this answer.")


def clear_cache() -> None:
    
    get_semantic_cache().clear()
    logger.info("Semantic cache cleared.")
