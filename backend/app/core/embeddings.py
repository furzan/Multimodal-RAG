import logging
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import settings
logger = logging.getLogger(__name__)
_embeddings: GoogleGenerativeAIEmbeddings | None = None


def _get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    
    global _embeddings
    
    if _embeddings is None:
    
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
    
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
            output_dimensionality= settings.pinecone_embedding_dim
        )
        
    return _embeddings


def embed_text(text: str) -> list[float]:
    
    model = _get_embeddings_model()
    return model.embed_query(text)


def embed_texts(texts: list[str]) -> list[list[float]]:
    
    if not texts:
        return []
    model = _get_embeddings_model()
    
    return model.embed_documents(texts)


def embed_query(query: str) -> list[float]:
    
    model = _get_embeddings_model()
    return model.embed_query(query)
