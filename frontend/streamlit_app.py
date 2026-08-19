"""
Streamlit frontend for the Multimodal RAG app.

Run with:
    uv run streamlit run streamlit_app.py

Single-page chat layout with a sidebar for document management.
All backend data is stored in session_state — zero API calls on
normal widget interactions (typing, clicking, changing top-k).
"""

import streamlit as st

from utils.api_client import (
    APIError,
    chat,
    delete_document,
    get_chunk_image_url,
    get_document_status,
    upload_document,
    list_documents,
    check_backend_health,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Multimodal RAG", page_icon="🌊", layout="wide")

st.markdown(
    """<style>
    .block-container { padding-top: 1.5rem !important; }
    .pill { display:inline-block; padding:2px 10px; border-radius:999px;
            font-size:.82rem; font-weight:600; }
    .pill-done   { background:#1B3C34; color:#2E8B77; }
    .pill-processing { background:#3A2E0A; color:#D4A72C; }
    .pill-pending { background:#2A2A2A; color:#999; }
    .pill-failed  { background:#3A1616; color:#D44; }
    </style>""",
    unsafe_allow_html=True,
)

# ── Session state defaults (run once, never again) ───────────────────────────

if "initialized" not in st.session_state:
    st.session_state.initialized = False
    st.session_state.backend_ok = False
    st.session_state.documents = []
    st.session_state.chat_history = []

# One-time health check on first load
if not st.session_state.initialized:
    st.session_state.backend_ok = check_backend_health()
    if st.session_state.backend_ok:
        try:
            st.session_state.documents = list_documents()
        except APIError:
            st.session_state.documents = []
    st.session_state.initialized = True

if not st.session_state.backend_ok:
    st.error(
        "⚠️ Can't reach the backend. Start it with "
        "`uvicorn app.main:app --reload` then refresh this page."
    )
    st.stop()

# ── Sidebar: documents ──────────────────────────────────────────────────────

with st.sidebar:
    st.header("📄 Documents")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_file is not None and st.button("Upload & Process", type="primary", use_container_width=True):
        with st.spinner("Uploading…"):
            try:
                result = upload_document(uploaded_file.getvalue(), uploaded_file.name)
                st.success(f"'{result['filename']}' — processing started.")
                # Refresh doc list after upload
                try:
                    st.session_state.documents = list_documents()
                except APIError:
                    pass
            except APIError as exc:
                st.error(f"Upload failed: {exc}")

    st.divider()

    if st.button("🔄 Refresh", use_container_width=True):
        try:
            st.session_state.documents = list_documents()
        except APIError as exc:
            st.error(f"Failed: {exc}")
        st.rerun()

    docs = st.session_state.documents
    if not docs:
        st.info("No documents yet.")
    else:
        pill_map = {"done": "pill-done", "processing": "pill-processing",
                     "pending": "pill-pending", "failed": "pill-failed"}
        for doc in docs:
            css = pill_map.get(doc["status"], "pill-pending")
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(
                    f'**{doc["filename"]}** <span class="pill {css}">{doc["status"]}</span>',
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🗑", key=f"del_{doc['document_id']}"):
                    try:
                        delete_document(doc["document_id"])
                        st.session_state.documents = list_documents()
                        st.rerun()
                    except APIError as exc:
                        st.error(str(exc))

# ── Main: Chat ───────────────────────────────────────────────────────────────

st.title("🌊 Multimodal RAG")

# Only show "done" docs in scope selector — no need to filter processing ones
done_docs = [d for d in st.session_state.documents if d["status"] == "done"]
scope_map: dict[str, str | None] = {"All documents": None}
scope_map.update({d["filename"]: d["document_id"] for d in done_docs})

c1, c2 = st.columns([4, 1])
with c1:
    selected_label = st.selectbox("Scope", list(scope_map.keys()), label_visibility="collapsed")
    selected_doc_id = scope_map[selected_label]
with c2:
    top_k = st.number_input("k", min_value=1, max_value=15, value=5, label_visibility="collapsed")

if not done_docs:
    st.info("Upload a PDF in the sidebar to get started.")


def _show_chunks(chunks: list[dict]) -> None:
    with st.expander(f"📎 {len(chunks)} chunk(s)"):
        for c in chunks:
            st.markdown(f"**{c['chunk_type'].capitalize()}** — p.{c.get('page_number', '?')}")
            if c["chunk_type"] == "image":
                st.image(get_chunk_image_url(c["chunk_id"]), width=280)
            else:
                st.text(c.get("preview", ""))
            st.divider()


for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("cache_hit"):
                st.caption("⚡ From cache")
            if msg.get("chunks"):
                _show_chunks(msg["chunks"])

query = st.chat_input("Ask a question…")

if query:
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                result = chat(query, document_id=selected_doc_id, top_k=top_k)
                st.markdown(result["answer"])
                if result.get("cache_hit"):
                    st.caption("⚡ From cache")
                chunks = result.get("retrieved_chunks") or []
                if chunks:
                    _show_chunks(chunks)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "cache_hit": result.get("cache_hit", False),
                    "chunks": chunks,
                })
            except APIError as exc:
                err = f"Error: {exc}"
                st.error(err)
                st.session_state.chat_history.append({"role": "assistant", "content": err})

if st.session_state.chat_history:
    if st.button("🗑 Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
