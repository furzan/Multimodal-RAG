import base64
import logging
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.config import settings
from app.core.retriever import RetrievedChunk
from app.models.db_models import ChunkType
logger = logging.getLogger(__name__)

_generation_llm: ChatGoogleGenerativeAI | ChatGroq | None = None


def _get_generation_llm() -> ChatGoogleGenerativeAI:
    
    global _generation_llm
    
    if _generation_llm is None:
    
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    
        _generation_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_vision_model,
            google_api_key=settings.google_api_key,
            temperature=0.2,
        )
    return _generation_llm


_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a set of "
    "documents. You will be given retrieved excerpts — which may include "
    "plain text, HTML tables, and images — as context. Answer the user's "
    "question using ONLY this context. If the context doesn't contain "
    "enough information to answer, say so plainly rather than guessing. "
    "When you rely on a table or image, mention that explicitly."
)


def _guess_mime_type(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
    }.get(ext, "image/png")


def _build_context_message_content(chunks: list[RetrievedChunk]) -> list[dict]:
    
    content: list[dict] = [{"type": "text", "text": "Retrieved context:\n"}]

    for i, chunk in enumerate(chunks, start=1):
        
        label = f"\n--- Context {i} (page {chunk.page_number or 'unknown'}, type={chunk.chunk_type.value}) ---\n"

        if chunk.chunk_type in (ChunkType.TEXT, ChunkType.TABLE):
            content.append({"type": "text", "text": label + (chunk.content_text or "")})

        elif chunk.chunk_type == ChunkType.IMAGE:
            if not chunk.content_path or not Path(chunk.content_path).exists():
                logger.warning("Image chunk %s has no valid file on disk; skipping.", chunk.chunk_id)
                continue
            image_bytes = Path(chunk.content_path).read_bytes()
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = _guess_mime_type(chunk.content_path)

            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": f"data:{mime_type};base64,{image_b64}",
                    # "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                }
            )

    return content


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:
    
    if not chunks:
        return (
            "I couldn't find any relevant information in the ingested "
            "documents to answer that question."
        )

    llm = _get_generation_llm()

    context_content = _build_context_message_content(chunks)
    context_content.append({"type": "text", "text": f"\n\nQuestion: {query}"})

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=context_content),
    ]

    response = llm.invoke(messages)
    return response.content.strip()
