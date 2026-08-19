import logging
from langchain_ollama import ChatOllama
from app.config import settings
from app.core.retriever import RetrievedChunk
from app.models.db_models import ChunkType

logger = logging.getLogger(__name__)

_generation_llm: ChatOllama | None = None


def _get_generation_llm() -> ChatOllama:
    """Singleton Ollama LLM used for answer generation."""
    global _generation_llm
    if _generation_llm is None:
        _generation_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0.2,
        )
    return _generation_llm


_SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions about a set of "
    "documents. You will be given retrieved excerpts — which may include "
    "plain text, HTML tables, and image descriptions — as context. Answer "
    "the user's question using ONLY this context. If the context doesn't "
    "contain enough information to answer, say so plainly rather than "
    "guessing. When you rely on a table or image description, mention that "
    "explicitly."
)


def _build_context(chunks: list[RetrievedChunk]) -> str:
    """Build a plain-text context string from retrieved chunks.

    For image chunks we use the pre-generated summary (created during
    ingestion by the Gemini vision model) instead of sending raw image
    bytes.  This lets us use a text-only LLM for generation.
    """
    parts: list[str] = ["Retrieved context:\n"]

    for i, chunk in enumerate(chunks, start=1):
        label = f"\n--- Context {i} (page {chunk.page_number or 'unknown'}, type={chunk.chunk_type.value}) ---\n"

        if chunk.chunk_type in (ChunkType.TEXT, ChunkType.TABLE):
            parts.append(label + (chunk.content_text or ""))

        elif chunk.chunk_type == ChunkType.IMAGE:
            # Use the summary that was generated at ingestion time
            description = chunk.summary or "[image — no description available]"
            parts.append(label + f"[Image description]: {description}")

    return "\n".join(parts)


def _response_text(content: str | list) -> str:
    if isinstance(content, str):
        return content.strip()

    text_parts = []
    for block in content:
        if isinstance(block, str):
            text_parts.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            text_parts.append(block["text"])

    return "".join(text_parts).strip()


def generate_answer(query: str, chunks: list[RetrievedChunk]) -> str:

    if not chunks:
        return (
            "I couldn't find any relevant information in the ingested "
            "documents to answer that question."
        )

    llm = _get_generation_llm()

    context = _build_context(chunks)
    prompt = f"{_SYSTEM_PROMPT}\n\n{context}\n\nQuestion: {query}"

    response = llm.invoke(prompt)
    return _response_text(response.content)
