"""
Streamlit frontend for the Multimodal RAG app.

Run with:
    uv run streamlit run streamlit_app.py

Kept deliberately simple: one file, two tabs (Upload, Chat), light theme
with a sea green accent. Theme colors also live in .streamlit/config.toml
so widgets Streamlit renders natively (buttons, inputs, spinners) match too.
"""

import time

import streamlit as st

from utils.api_client import (
    APIError,
    chat,
    check_backend_health,
    delete_document,
    get_chunk_image_url,
    get_document_status,
    list_documents,
    upload_document,
)

SEA_GREEN = "#2E8B77"
SEA_GREEN_DARK = "#1F6B5C"
SEA_GREEN_LIGHT = "#E6F3EF"
INK = "#1C2B27"

st.set_page_config(page_title="Multimodal RAG", page_icon="🌊", layout="wide")

# --- Minimal custom CSS: sea green accents on a light background ---
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #FAFDFC;
        }}
        h1, h2, h3 {{
            color: {INK};
        }}
        /* Primary buttons */
        .stButton > button[kind="primary"] {{
            background-color: {SEA_GREEN};
            border-color: {SEA_GREEN};
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {SEA_GREEN_DARK};
            border-color: {SEA_GREEN_DARK};
        }}
        /* Secondary buttons */
        .stButton > button {{
            border-color: {SEA_GREEN};
            color: {SEA_GREEN_DARK};
        }}
        .stButton > button:hover {{
            border-color: {SEA_GREEN_DARK};
            color: {SEA_GREEN_DARK};
        }}
        /* Tabs */
        .stTabs [aria-selected="true"] {{
            color: {SEA_GREEN_DARK};
            border-bottom-color: {SEA_GREEN} !important;
        }}
        /* Status pill */
        .status-pill {{
            display: inline-block;
            padding: 2px 10px;
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        .status-done {{ background-color: {SEA_GREEN_LIGHT}; color: {SEA_GREEN_DARK}; }}
        .status-processing {{ background-color: #FFF4D6; color: #92700C; }}
        .status-pending {{ background-color: #F0F0F0; color: #666; }}
        .status-failed {{ background-color: #FBE7E7; color: #A33131; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌊 Multimodal RAG")
st.caption("Ask questions about your PDFs — including their tables and images.")

if not check_backend_health():
    st.error(
        "Can't reach the backend API. Start it with `uvicorn app.main:app --reload` "
        "in the backend directory, then refresh this page."
    )
    st.stop()

tab_chat, tab_upload = st.tabs(["💬 Chat", "📄 Documents"])

# ============================== CHAT TAB ==============================
with tab_chat:
    try:
        documents = list_documents()
    except APIError as exc:
        st.error(f"Could not load documents: {exc}")
        documents = []

    done_documents = [d for d in documents if d["status"] == "done"]

    col_scope, col_topk = st.columns([3, 1])
    with col_scope:
        scope_options = {"All documents": None}
        scope_options.update({d["filename"]: d["document_id"] for d in done_documents})
        selected_label = st.selectbox("Search scope", options=list(scope_options.keys()))
        selected_document_id = scope_options[selected_label]
    with col_topk:
        top_k = st.number_input("Chunks to retrieve", min_value=1, max_value=15, value=5)

    if not done_documents:
        st.info("No processed documents yet. Upload one in the **Documents** tab.")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    def render_chunks(chunks: list[dict]) -> None:
        with st.expander(f"📎 {len(chunks)} retrieved chunk(s)"):
            for c in chunks:
                st.markdown(f"**{c['chunk_type'].capitalize()}** — page {c.get('page_number', 'unknown')}")
                if c["chunk_type"] == "image":
                    st.image(get_chunk_image_url(c["chunk_id"]), width=280)
                else:
                    st.text(c["preview"])
                st.divider()

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("cache_hit"):
                    st.caption("⚡ Answered from cache")
                if msg.get("chunks"):
                    render_chunks(msg["chunks"])

    query = st.chat_input("Ask a question about your documents...")

    if query:
        st.session_state["chat_history"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = chat(query, document_id=selected_document_id, top_k=top_k)
                    st.markdown(result["answer"])
                    if result.get("cache_hit"):
                        st.caption("⚡ Answered from cache")
                    chunks = result.get("retrieved_chunks") or []
                    if chunks:
                        render_chunks(chunks)
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "cache_hit": result.get("cache_hit", False),
                            "chunks": chunks,
                        }
                    )
                except APIError as exc:
                    error_text = f"Something went wrong: {exc}"
                    st.error(error_text)
                    st.session_state["chat_history"].append({"role": "assistant", "content": error_text})

    if st.session_state["chat_history"]:
        if st.button("Clear chat"):
            st.session_state["chat_history"] = []
            st.rerun()

# ============================== DOCUMENTS TAB ==============================
with tab_upload:
    st.subheader("Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None and st.button("Upload & Process", type="primary"):
        with st.spinner("Uploading..."):
            try:
                result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
                st.session_state["last_uploaded_document_id"] = result["document_id"]
                st.success(f"Uploaded '{result['filename']}' — processing started.")
            except APIError as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()
    st.subheader("Your documents")

    if st.button("Refresh"):
        st.rerun()

    try:
        documents = list_documents()
    except APIError as exc:
        st.error(f"Could not load documents: {exc}")
        documents = []

    if not documents:
        st.info("No documents uploaded yet.")
    else:
        status_classes = {
            "done": "status-done",
            "processing": "status-processing",
            "pending": "status-pending",
            "failed": "status-failed",
        }

        for doc in documents:
            c1, c2, c3 = st.columns([5, 2, 1])
            with c1:
                st.write(f"**{doc['filename']}**")
                st.caption(doc["document_id"])
            with c2:
                css_class = status_classes.get(doc["status"], "status-pending")
                st.markdown(
                    f'<span class="status-pill {css_class}">{doc["status"]}</span>',
                    unsafe_allow_html=True,
                )
                if doc["status"] == "failed":
                    try:
                        status = get_document_status(doc["document_id"])
                        st.caption(status.get("error_message") or "Unknown error")
                    except APIError:
                        pass
            with c3:
                if st.button("Delete", key=f"del_{doc['document_id']}"):
                    try:
                        delete_document(doc["document_id"])
                        st.rerun()
                    except APIError as exc:
                        st.error(str(exc))

    # Auto-poll while the most recently uploaded doc is still processing.
    last_id = st.session_state.get("last_uploaded_document_id")
    if last_id:
        try:
            status = get_document_status(last_id)
            if status["status"] in ("pending", "processing"):
                st.info(f"Processing '{status['filename']}'... refreshing in 3s.")
                time.sleep(3)
                st.rerun()
            elif status["status"] == "done":
                del st.session_state["last_uploaded_document_id"]
        except APIError:
            del st.session_state["last_uploaded_document_id"]
