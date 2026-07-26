"""Streamlit UI for the domain-specific RAG chatbot.

The user experience is intentionally simple: upload PDFs, ingest them into a
local vector store, ask questions in chat, and inspect the retrieved sources
used to build the response.
"""

from __future__ import annotations

from datetime import datetime
import time

import streamlit as st

from config import DATA_DIR, EMBEDDING_MODEL_NAME
from generator import generate_answer
from ingest import get_collection, get_document_stats, get_indexed_documents, ingest_pdfs
from retriever import retrieve_context


st.set_page_config(page_title="Domain Knowledge Assistant", page_icon="🤖", layout="wide")


def _format_preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _similarity_color(similarity: object) -> str:
    try:
        value = float(similarity or 0)
    except (TypeError, ValueError):
        return "#ef4444"
    if value >= 0.7:
        return "#22c55e"
    if value >= 0.4:
        return "#f59e0b"
    if value >= 0.2:
        return "#f97316"
    return "#ef4444"


def _get_average_similarity(messages: list[dict[str, object]]) -> float | None:
    values: list[float] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for source in message.get("sources") or []:
            try:
                values.append(float(source.get("similarity", 0) or 0))
            except (TypeError, ValueError):
                continue
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _get_last_response_time(messages: list[dict[str, object]]) -> int | None:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("response_time_ms") is not None:
            return int(message["response_time_ms"])
    return None


def _render_stat_card(icon: str, title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-icon">{icon}</div>
            <div class="stat-title">{title}</div>
            <div class="stat-value">{value}</div>
            <div class="stat-meta">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_document_cards() -> list[dict[str, object]]:
    collection = get_collection()
    result = collection.get(include=["metadatas"])
    metadatas = result.get("metadatas") or []
    rows: dict[str, dict[str, object]] = {}

    for metadata in metadatas:
        if isinstance(metadata, list):
            items = metadata
        elif isinstance(metadata, dict):
            items = [metadata]
        else:
            continue

        for item in items:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            if not source:
                continue
            entry = rows.setdefault(
                str(source),
                {"filename": str(source), "pages": set(), "chunks": 0, "status": "Indexed"},
            )
            entry["pages"].add(item.get("page"))
            entry["chunks"] = int(entry.get("chunks", 0)) + 1

    cards = []
    for source, entry in rows.items():
        pages = entry["pages"]
        cards.append(
            {
                "filename": source,
                "pages": len(pages),
                "chunks": int(entry.get("chunks", 0)),
                "status": entry.get("status", "Indexed"),
            }
        )
    cards.sort(key=lambda item: str(item["filename"]))
    return cards


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    :root {
        --bg: #09090b;
        --surface: #111114;
        --surface-2: #17171c;
        --border: rgba(255, 255, 255, 0.08);
        --text: #f5f5f6;
        --muted: #8d94a3;
        --accent: #6366f1;
        --accent-soft: rgba(99, 102, 241, 0.16);
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background: var(--bg);
        color: var(--text);
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }
    .stSidebar {
        background: var(--bg);
        border-right: 1px solid var(--border);
        padding-top: 0.8rem;
    }
    .st-emotion-cache-1y4p8pa {
        background: var(--bg);
    }
    .page-header {
        margin-bottom: 1.2rem;
        padding-top: 0.2rem;
    }
    .page-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--text);
        letter-spacing: -0.01em;
    }
    .page-subtitle {
        margin-top: 0.2rem;
        font-size: 0.93rem;
        color: var(--muted);
        line-height: 1.5;
    }
    .sidebar-block {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.8rem;
        background: var(--surface);
    }
    .sidebar-title {
        font-size: 0.98rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.2rem;
    }
    .sidebar-copy {
        font-size: 0.86rem;
        color: var(--muted);
        line-height: 1.55;
    }
    .sidebar-section-title {
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.55rem;
    }
    .doc-item {
        display: flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.4rem 0;
        color: var(--text);
        font-size: 0.92rem;
    }
    .doc-item .doc-check {
        color: var(--accent);
        flex-shrink: 0;
    }
    .empty-state {
        min-height: 320px;
        display: flex;
        align-items: center;
        justify-content: center;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 2rem;
        color: var(--muted);
        text-align: center;
        background: var(--surface);
    }
    .empty-title {
        font-size: 0.98rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.35rem;
    }
    .empty-copy {
        font-size: 0.92rem;
        line-height: 1.6;
        max-width: 520px;
    }
    .message-stack {
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    .message-row {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        max-width: 860px;
    }
    .message-row.user {
        margin-left: auto;
        justify-content: flex-end;
    }
    .message-row.assistant {
        margin-right: auto;
    }
    .message-avatar {
        width: 1.8rem;
        height: 1.8rem;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
        flex-shrink: 0;
        border: 1px solid var(--border);
        background: var(--surface-2);
        color: var(--text);
    }
    .message-bubble {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.8rem 0.95rem;
        line-height: 1.7;
        font-size: 0.95rem;
        color: var(--text);
        background: transparent;
    }
    .message-row.user .message-bubble {
        background: var(--surface-2);
        border-color: rgba(99, 102, 241, 0.18);
    }
    .message-meta {
        color: var(--muted);
        font-size: 0.76rem;
        margin: 0 0 0.35rem 0;
    }
    .typing-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0.55rem 0.7rem;
        border-radius: 999px;
        background: var(--surface-2);
        border: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.92rem;
        margin-bottom: 0.4rem;
    }
    .typing-dots {
        display: inline-flex;
        gap: 0.25rem;
    }
    .typing-dots span {
        width: 0.36rem;
        height: 0.36rem;
        border-radius: 50%;
        background: var(--accent);
        animation: blink 1.1s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) { animation-delay: 0.15s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.3s; }
    @keyframes blink {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.45; }
        40% { transform: scale(1); opacity: 1; }
    }
    .source-group {
        margin-top: 0.5rem;
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
    }
    .source-item {
        padding: 0.55rem 0.65rem;
        border-radius: 10px;
        background: var(--surface-2);
        border: 1px solid var(--border);
    }
    .source-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.6rem;
        font-size: 0.84rem;
        color: var(--text);
    }
    .source-meta {
        color: var(--muted);
        font-size: 0.8rem;
    }
    .source-preview {
        margin-top: 0.3rem;
        font-size: 0.82rem;
        line-height: 1.5;
        color: var(--muted);
    }
    .confidence-block {
        margin-top: 0.5rem;
        padding: 0.55rem 0.65rem;
        border-radius: 10px;
        background: var(--surface-2);
        border: 1px solid var(--border);
    }
    .confidence-row {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        margin-top: 0.1rem;
    }
    .confidence-label {
        color: var(--muted);
        font-size: 0.8rem;
        min-width: 72px;
    }
    .confidence-bar {
        flex: 1;
        height: 0.28rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: inherit;
    }
    .confidence-value {
        color: var(--text);
        font-size: 0.8rem;
        font-weight: 600;
    }
    .chat-input-shell {
        position: sticky;
        bottom: 0;
        background: var(--bg);
        padding-top: 0.6rem;
        z-index: 5;
    }
    .stChatInput > div {
        border: 1px solid var(--border) !important;
        border-radius: 999px !important;
        background: var(--surface) !important;
        box-shadow: none !important;
    }
    .stButton > button {
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--accent);
        color: white;
    }
    .stButton > button:hover {
        opacity: 0.95;
    }
    .stFileUploader > div {
        border: 1px dashed rgba(99, 102, 241, 0.32) !important;
        border-radius: 14px !important;
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <div class="page-header">
        <div class="page-title">Domain Knowledge Assistant</div>
        <div class="page-subtitle">Grounded answers from your uploaded documents.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-block">
            <div class="sidebar-title">Domain Knowledge Assistant</div>
            <div class="sidebar-copy">Ask questions about your documents.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("<div class='sidebar-copy'>PDFs are saved locally and indexed for retrieval.</div>", unsafe_allow_html=True)
    if st.button("Upload PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF before ingesting.")
        else:
            persisted_files = []
            for uploaded in uploaded_files:
                destination = DATA_DIR / uploaded.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(uploaded.getvalue())
                persisted_files.append(destination)

            with st.spinner("Reading documents..."):
                result = ingest_pdfs(persisted_files)
            if result["ingested"]:
                st.success(f"Indexed {len(result['ingested'])} document(s) and {result['chunk_count']} chunk(s).")
            if result["skipped"]:
                st.info("Skipped already indexed files: " + ", ".join(result["skipped"]))
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-section-title'>Library</div>", unsafe_allow_html=True)
    documents = _get_document_cards()
    if documents:
        for document in documents:
            st.markdown(
                f"""
                <div class="doc-item">
                    <span class="doc-check">✓</span>
                    <span>{document['filename']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No documents have been indexed yet.")
    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div>
                <div class="empty-title">Start by uploading a PDF.</div>
                <div class="empty-copy">Ask a question in natural language and receive a grounded answer with source references.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="message-stack">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message.get("role", "assistant")
    is_user = role == "user"
    avatar = "🧑" if is_user else "🤖"
    bubble_class = "user" if is_user else "assistant"
    with st.container():
        if message.get("timestamp"):
            st.markdown(f'<div class="message-meta">{message["timestamp"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="message-row {bubble_class}"><div class="message-avatar">{avatar}</div><div class="message-bubble">{message["content"]}</div></div>',
            unsafe_allow_html=True,
        )
        if not is_user and message.get("sources"):
            top_similarity = message["sources"][0].get("similarity", 0) if message.get("sources") else None
            if top_similarity is not None:
                percentage = max(6, int(float(top_similarity) * 100))
                color = _similarity_color(top_similarity)
                st.markdown(
                    f'<div class="message-row assistant"><div class="message-avatar">✦</div><div class="source-group"><div class="confidence-block"><div class="source-head"><span>Confidence</span><span class="source-meta">{int(float(top_similarity) * 100)}%</span></div><div class="confidence-row"><div class="confidence-bar"><div class="confidence-fill" style="width:{percentage}%; background:{color};"></div></div><div class="confidence-value">{int(float(top_similarity) * 100)}%</div></div></div>',
                    unsafe_allow_html=True,
                )
            for source in message["sources"]:
                preview = _format_preview(str(source.get("content", "")))
                st.markdown(
                    f"""
                    <div class="message-row assistant">
                        <div class="message-avatar">📄</div>
                        <div class="source-item">
                            <div class="source-head">
                                <span>{source.get('source', 'unknown')}</span>
                                <span class="source-meta">p{source.get('page', 'unknown')}</span>
                            </div>
                            <div class="source-preview">{preview}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if top_similarity is not None:
                st.markdown("</div></div>", unsafe_allow_html=True)
        elif not is_user and message.get("is_grounded") is False:
            st.caption("No relevant context was found, so the answer is not grounded in the uploaded documents.")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="chat-input-shell">', unsafe_allow_html=True)
prompt = st.chat_input("Ask anything about your uploaded documents...", key="rag_chat_input")
st.markdown('</div>', unsafe_allow_html=True)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M")})
    with st.container():
        st.markdown(f'<div class="message-meta">{datetime.now().strftime("%H:%M")}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="message-row user"><div class="message-avatar">🧑</div><div class="message-bubble">{prompt}</div></div>',
            unsafe_allow_html=True,
        )

    with st.container():
        st.markdown('<div class="typing-pill"><div class="typing-dots"><span></span><span></span><span></span></div><div>Searching documents…</div></div>', unsafe_allow_html=True)
        start_time = time.perf_counter()
        with st.spinner("Retrieving relevant chunks..."):
            context_chunks = retrieve_context(prompt)
        if not context_chunks:
            answer = "I do not have enough relevant context to answer that question confidently."
            sources = []
            is_grounded = False
            st.info("No relevant context was found, so the answer is not grounded in the uploaded documents.")
        else:
            st.markdown('<div class="typing-pill"><div class="typing-dots"><span></span><span></span><span></span></div><div>Generating answer…</div></div>', unsafe_allow_html=True)
            with st.spinner("Generating grounded answer..."):
                try:
                    response = generate_answer(prompt, context_chunks)
                    answer = response["answer"]
                    sources = response["sources"]
                    is_grounded = bool(response.get("grounded", False))
                except Exception as exc:
                    answer = "The answer could not be generated right now. Please check the Gemini setup and try again."
                    sources = []
                    is_grounded = False
                    st.error(str(exc))

        response_time_ms = int((time.perf_counter() - start_time) * 1000)
        st.markdown(
            f'<div class="message-row assistant"><div class="message-avatar">🤖</div><div class="message-bubble">{answer}</div></div>',
            unsafe_allow_html=True,
        )
        if sources:
            top_similarity = sources[0].get("similarity", 0) if sources else None
            if top_similarity is not None:
                percentage = max(6, int(float(top_similarity) * 100))
                color = _similarity_color(top_similarity)
                st.markdown(
                    f'<div class="message-row assistant"><div class="message-avatar">✦</div><div class="source-group"><div class="confidence-block"><div class="source-head"><span>Confidence</span><span class="source-meta">{int(float(top_similarity) * 100)}%</span></div><div class="confidence-row"><div class="confidence-bar"><div class="confidence-fill" style="width:{percentage}%; background:{color};"></div></div><div class="confidence-value">{int(float(top_similarity) * 100)}%</div></div></div>',
                    unsafe_allow_html=True,
                )
            for source in sources:
                preview = _format_preview(str(source.get("content", "")))
                st.markdown(
                    f"""
                    <div class="message-row assistant">
                        <div class="message-avatar">📄</div>
                        <div class="source-item">
                            <div class="source-head">
                                <span>{source.get('source', 'unknown')}</span>
                                <span class="source-meta">p{source.get('page', 'unknown')}</span>
                            </div>
                            <div class="source-preview">{preview}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if top_similarity is not None:
                st.markdown("</div></div>", unsafe_allow_html=True)
        elif not is_grounded:
            st.caption("No sources were used because no relevant context was found.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "is_grounded": is_grounded,
            "timestamp": datetime.now().strftime("%H:%M"),
            "response_time_ms": response_time_ms,
        }
    )
    st.rerun()
