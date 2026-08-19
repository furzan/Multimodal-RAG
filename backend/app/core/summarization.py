import base64
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from app.config import settings
logger = logging.getLogger(__name__)

_ollama_llm: ChatOllama | None = None
_gemini_vision_llm: ChatGoogleGenerativeAI | None = None


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


def _get_ollama_llm() -> ChatOllama:
    global _ollama_llm
    if _ollama_llm is None:
        _ollama_llm = ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    return _ollama_llm


def _get_gemini_vision_llm() -> ChatGoogleGenerativeAI:
    global _gemini_vision_llm
    if _gemini_vision_llm is None:
        if not settings.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        _gemini_vision_llm = ChatGoogleGenerativeAI(
            model=settings.gemini_vision_model,
            google_api_key=settings.google_api_key,
            temperature=0,
        )
    return _gemini_vision_llm


_TEXT_SUMMARY_PROMPT = (
    "You are creating a compact summary of a document excerpt for a search "
    "index. The summary will be embedded and used to retrieve this excerpt "
    "later, so focus on the key facts, entities, and topics — not style. "
    "Do not add commentary or preamble. Excerpt:\n\n{content}"
)

_TABLE_SUMMARY_PROMPT = (
    "You are creating a compact summary of a table (given as HTML) for a "
    "search index. Describe what the table contains: its columns, the kind "
    "of data in it, and any notable values or trends. The summary will be "
    "embedded for retrieval, so be specific and factual, not stylistic. "
    "Do not add commentary or preamble. Table HTML:\n\n{content}"
)

_IMAGE_SUMMARY_PROMPT = (
    "You are creating a compact summary of an image extracted from a "
    "document (could be a chart, diagram, photo, or screenshot) for a "
    "search index. Describe what the image shows in enough detail that "
    "someone could find it via a text search — mention any visible labels, "
    "numbers, or trends if it's a chart/diagram. Do not add commentary or "
    "preamble."
)


def summarize_text(content: str) -> str:

    llm = _get_ollama_llm()
    response = llm.invoke(_TEXT_SUMMARY_PROMPT.format(content=content))
    return _response_text(response.content)


def summarize_table(table_html: str) -> str:

    llm = _get_ollama_llm()
    response = llm.invoke(_TABLE_SUMMARY_PROMPT.format(content=table_html))
    return _response_text(response.content)


def summarize_image(image_bytes: bytes, mime_type: str = "image/png") -> str:
    
    llm = _get_gemini_vision_llm()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": _IMAGE_SUMMARY_PROMPT},
            {
                "type": "image_url",
                "image_url": f"data:{mime_type};base64,{image_b64}",
                # "image_url": {"url": f"data:{mime_type};base64,{image_b64}"}, 
            },
        ]
    )
    response = llm.invoke([message])
    return _response_text(response.content)
