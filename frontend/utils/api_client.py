"""
Thin wrapper around the FastAPI backend's HTTP API. Keeping all requests
here (rather than scattering requests.get/post calls across pages) means
the base URL, error handling, and response shapes only need to be defined
once.
"""

import os

import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")


class APIError(Exception):
    """Raised when the backend returns a non-2xx response, with the detail message."""


def _handle_response(response: requests.Response) -> dict:
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIError(f"[{response.status_code}] {detail}")
    return response.json()


def upload_document(file_bytes: bytes, filename: str) -> dict:
    """Uploads a PDF. Returns {document_id, filename, status, message}."""
    response = requests.post(
        f"{BACKEND_URL}/ingest/upload",
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=60,
    )
    return _handle_response(response)


def list_documents() -> list[dict]:
    response = requests.get(f"{BACKEND_URL}/documents", timeout=15)
    return _handle_response(response)["documents"]


def get_document_status(document_id: str) -> dict:
    response = requests.get(f"{BACKEND_URL}/documents/{document_id}/status", timeout=15)
    return _handle_response(response)


def delete_document(document_id: str) -> dict:
    response = requests.delete(f"{BACKEND_URL}/documents/{document_id}", timeout=30)
    return _handle_response(response)


def chat(query: str, document_id: str | None = None, top_k: int = 5) -> dict:
    """Sends a chat query. Returns {answer, cache_hit, retrieved_chunks}."""
    payload = {"query": query, "top_k": top_k}
    if document_id:
        payload["document_id"] = document_id

    # Generation can be slow (multimodal LLM call over several images),
    # so this timeout is generous relative to the other endpoints.
    response = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=120)
    return _handle_response(response)


def get_chunk_image_url(chunk_id: str) -> str:
    """Returns a direct URL to a retrieved image chunk (for st.image)."""
    return f"{BACKEND_URL}/chat/image/{chunk_id}"


def check_backend_health() -> bool:
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(ttl=5)
def cached_list_documents() -> list[dict]:
    """Cached wrapper for polling document status without hammering the backend."""
    return list_documents()
