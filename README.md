# Multimodal RAG

A FastAPI and Streamlit application for asking questions about PDF documents containing text, tables, and images. The application combines document partitioning, multimodal summarization, vector retrieval, and semantic caching to provide grounded answers with source chunks.

## Features

- Upload and process PDF documents asynchronously
- Extract text, tables, and images from documents
- Generate summaries and embeddings for multimodal chunks
- Retrieve relevant context with Pinecone
- Generate answers with the configured LLM providers
- Cache similar queries with Redis
- Display retrieved source chunks, including extracted images

## Prerequisites

- Python 3.11 or later
- [`uv`](https://docs.astral.sh/uv/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [Poppler](https://poppler.freedesktop.org/)
- Redis running locally or remotely
- Credentials for the providers configured in `.env` (for example, Pinecone, Gemini, Groq, or Ollama)

Tesseract and Poppler are system dependencies used by the PDF and image extraction pipeline. Make sure their executables are available on your `PATH`.

## Installation

From the repository root, sync the backend environment:

```bash
uv sync
```

The frontend has its own project metadata. Its dependencies are installed automatically when the frontend command is run, or can be synced explicitly:

```bash
cd frontend
uv sync
cd ..
```

Create a `.env` file with the provider settings required by your selected models. The main configuration options include `PINECONE_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY`, `OLLAMA_BASE_URL`, and `REDIS_URL`.

## Run The Application

Start the backend from the `backend` directory:

```bash
cd backend
uv run uvicorn app.main:app --reload
```

In a second terminal, start the frontend from the `frontend` directory:

```bash
cd frontend
uv run streamlit run streamlit_app.py
```

Open the Streamlit URL shown in the terminal, usually `http://localhost:8501`.

## Processing Flow

1. A user uploads a PDF through the Streamlit interface.
2. The FastAPI backend saves the file and queues background ingestion.
3. The PDF is partitioned into text, table, and image chunks. Images are stored for later display.
4. Each chunk is summarized using the configured text, table, or vision model.
5. Chunk summaries are embedded and indexed in Pinecone. Original chunk data and metadata are stored in SQLite.
6. A user question is checked against the Redis semantic cache.
7. If there is no cache hit, the retriever finds relevant chunks from Pinecone, and the RAG chain generates an answer from that context.
8. The answer and retrieved sources are returned to Streamlit and shown in the chat interface.

## Project Structure

```text
backend/   FastAPI API, ingestion pipeline, retrieval, storage, and model integrations
frontend/  Streamlit user interface and backend API client
uploads/   Local upload directory
output/    Example or generated processing output
```

## API Health Check

Once the backend is running, verify it with:

```text
http://localhost:8000/health
```
