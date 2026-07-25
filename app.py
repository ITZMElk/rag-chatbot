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
        --bg: #09090B;
        --surface: #131722;
        --surface-2: #171C2A;
        --surface-3: #1A2130;
        --accent: #6366F1;
        --accent-2: #06B6D4;
        --text: #F8FAFC;
        --muted: #94A3B8;
        --border: rgba(255,255,255,0.08);
        --shadow: 0 12px 30px rgba(2, 6, 23, 0.25);
    }
    .stApp {
        background: var(--bg);
        color: var(--text);
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    .stSidebar {
        background: linear-gradient(180deg, rgba(19, 23, 34, 0.98), rgba(9, 9, 11, 0.98));
        border-right: 1px solid var(--border);
    }
    .hero-card {
        background: linear-gradient(120deg, rgba(99, 102, 241, 0.18), rgba(6, 182, 212, 0.12));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.25rem 1.35rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text);
        margin-bottom: 0.35rem;
    }
    .hero-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.6;
    }
    .hero-description {
        margin-top: 0.55rem;
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.65;
        max-width: 760px;
    }
    .badge-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-bottom: 0.7rem;
    }
    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.65rem;
        border-radius: 999px;
        border: 1px solid rgba(99, 102, 241, 0.3);
        background: rgba(99, 102, 241, 0.12);
        color: #dbe4ff;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .stat-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        box-shadow: var(--shadow);
        transition: transform 180ms ease, border-color 180ms ease;
        min-height: 122px;
        margin-bottom: 0.8rem;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.35);
    }
    .stat-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 2.1rem;
        height: 2.1rem;
        border-radius: 10px;
        background: rgba(99, 102, 241, 0.16);
        margin-bottom: 0.45rem;
        color: var(--accent-2);
    }
    .stat-title {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.2rem;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 0.2rem;
    }
    .stat-meta {
        font-size: 0.78rem;
        color: var(--muted);
        line-height: 1.45;
    }
    .sidebar-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 0.85rem 0.95rem;
        margin-bottom: 0.8rem;
        box-shadow: var(--shadow);
    }
    .sidebar-intro {
        font-size: 0.9rem;
        color: var(--muted);
        line-height: 1.6;
        margin-top: 0.3rem;
    }
    .upload-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12), rgba(6, 182, 212, 0.09));
        border: 1px dashed rgba(99, 102, 241, 0.32);
        border-radius: 16px;
        padding: 0.9rem;
        margin-bottom: 0.9rem;
    }
    .upload-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.25rem;
    }
    .upload-copy {
        font-size: 0.82rem;
        color: var(--muted);
        line-height: 1.5;
        margin-bottom: 0.4rem;
    }
    .doc-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        margin-bottom: 0.55rem;
        transition: border-color 180ms ease, transform 180ms ease;
    }
    .doc-card:hover {
        border-color: rgba(99, 102, 241, 0.35);
        transform: translateY(-1px);
    }
    .doc-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
    }
    .doc-name {
        font-weight: 600;
        color: var(--text);
        font-size: 0.9rem;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .doc-meta {
        font-size: 0.76rem;
        color: var(--muted);
        display: flex;
        gap: 0.4rem;
        flex-wrap: wrap;
        margin-top: 0.35rem;
    }
    .doc-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        padding: 0.24rem 0.5rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.04);
        border: 1px solid var(--border);
    }
    .doc-delete {
        color: var(--muted);
        font-size: 0.9rem;
        cursor: default;
    }
    .empty-state {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.4rem;
        text-align: center;
        box-shadow: var(--shadow);
    }
    .empty-emoji {
        font-size: 2.2rem;
        margin-bottom: 0.35rem;
    }
    .empty-title {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
        color: var(--text);
    }
    .empty-copy {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.6;
    }
    .message-shell {
        margin: 0.2rem 0 0.7rem 0;
    }
    .chat-bubble {
        border-radius: 16px;
        padding: 0.8rem 0.95rem;
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
        line-height: 1.6;
        font-size: 0.95rem;
        margin: 0.2rem 0 0.4rem 0;
    }
    .user-bubble {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(6, 182, 212, 0.1));
        margin-left: 2rem;
        border: 1px solid rgba(99, 102, 241, 0.24);
    }
    .assistant-bubble {
        background: rgba(255,255,255,0.03);
        margin-right: 2rem;
        border-left: 3px solid rgba(99, 102, 241, 0.45);
    }
    .message-meta {
        color: var(--muted);
        font-size: 0.74rem;
        margin-bottom: 0.35rem;
    }
    .typing-card {
        display: inline-flex;
        align-items: center;
        gap: 0.7rem;
        padding: 0.55rem 0.7rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        color: var(--muted);
        margin: 0.2rem 0 0.4rem 0;
    }
    .typing-dots {
        display: inline-flex;
        gap: 0.25rem;
    }
    .typing-dots span {
        width: 0.4rem;
        height: 0.4rem;
        border-radius: 50%;
        background: var(--accent-2);
        animation: blink 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(2) {
        animation-delay: 0.15s;
    }
    .typing-dots span:nth-child(3) {
        animation-delay: 0.3s;
    }
    @keyframes blink {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.45; }
        40% { transform: scale(1); opacity: 1; }
    }
    .citation-card {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.7rem 0.8rem;
        background: rgba(255,255,255,0.025);
        margin-top: 0.5rem;
        box-shadow: 0 8px 16px rgba(2, 6, 23, 0.14);
    }
    .citation-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.6rem;
        margin-bottom: 0.35rem;
    }
    .citation-title {
        font-weight: 600;
        color: var(--text);
        font-size: 0.87rem;
    }
    .citation-meta {
        font-size: 0.76rem;
        color: var(--muted);
    }
    .similarity-bar {
        height: 0.36rem;
        width: 100%;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        margin-top: 0.35rem;
    }
    .similarity-fill {
        height: 100%;
        border-radius: inherit;
    }
    .citation-preview {
        margin-top: 0.45rem;
        color: var(--muted);
        font-size: 0.82rem;
        line-height: 1.55;
    }
    .citation-highlight {
        display: inline-block;
        padding: 0.1rem 0.3rem;
        border-radius: 0.35rem;
        background: rgba(99, 102, 241, 0.14);
        color: #e2e8f0;
    }
    .chat-input-shell {
        position: sticky;
        bottom: 0;
        background: var(--bg);
        padding-top: 0.6rem;
        z-index: 5;
    }
    .stChatInput {
        position: sticky;
        bottom: 0;
    }
    .stChatInput > div {
        border: 1px solid var(--border) !important;
        border-radius: 999px !important;
        background: var(--surface) !important;
        box-shadow: var(--shadow);
    }
    .stButton > button {
        border-radius: 999px;
        border: 1px solid var(--border);
        background: linear-gradient(90deg, var(--accent), var(--accent-2));
        color: white;
    }
    .stButton > button:hover {
        border-color: rgba(255,255,255,0.15);
        opacity: 0.95;
    }
    .stFileUploader > div {
        border: 1px dashed rgba(99, 102, 241, 0.32) !important;
        border-radius: 14px !important;
        background: rgba(99, 102, 241, 0.06) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown(
    """
    <div class="hero-card">
        <div class="badge-row">
            <span class="badge">🧠 RAG</span>
            <span class="badge">✨ Gemini AI</span>
            <span class="badge">🗂️ ChromaDB</span>
            <span class="badge">📎 Source Citations</span>
        </div>
        <div class="hero-title">🧠 Domain Knowledge Assistant</div>
        <div class="hero-subtitle">Chat with your PDFs using Retrieval-Augmented Generation.</div>
        <div class="hero-description">Upload documents, search semantically, and receive grounded answers with page citations and confidence scores.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

stats = get_document_stats()
question_count = len([message for message in st.session_state.messages if message.get("role") == "user"])
average_similarity = _get_average_similarity(st.session_state.messages)
last_response_time = _get_last_response_time(st.session_state.messages)
embedding_model = EMBEDDING_MODEL_NAME

stat_items = [
    ("📄", "Documents", str(stats["document_count"]), "Indexed knowledge sources"),
    ("🧩", "Chunks", str(stats["chunk_count"]), "Stored retrieval units"),
    ("🧠", "Embedding Model", embedding_model, "Local semantic encoder"),
    ("❓", "Questions Asked", str(question_count), "Prompts in the current session"),
    ("📈", "Average Similarity", f"{average_similarity:.2f}" if average_similarity is not None else "—", "Across retrieved passages"),
    ("⏱️", "Response Time", f"{last_response_time} ms" if last_response_time is not None else "—", "Last answer latency"),
]
stat_columns = st.columns(3)
for index, (icon, title, value, subtitle) in enumerate(stat_items):
    with stat_columns[index % 3]:
        _render_stat_card(icon, title, value, subtitle)

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
            <div style="display:flex;align-items:center;gap:0.6rem;">
                <div style="font-size:1.2rem;">🧠</div>
                <div>
                    <div style="font-weight:700;color:#F8FAFC;">Domain Knowledge Assistant</div>
                    <div class="sidebar-intro">A lightweight AI knowledge base for your PDFs.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sidebar-card'><div class='upload-title'>Knowledge Base</div><div class='upload-copy'>Upload PDFs to build a searchable local knowledge base.</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    st.markdown("<div class='upload-copy'>PDFs are saved locally and indexed for retrieval.</div>", unsafe_allow_html=True)
    if st.button("Ingest PDFs", use_container_width=True):
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

    st.markdown("<div class='sidebar-card'><div class='upload-title'>Indexed documents</div></div>", unsafe_allow_html=True)
    documents = _get_document_cards()
    if documents:
        for document in documents:
            st.markdown(
                f"""
                <div class="doc-card">
                    <div class="doc-top">
                        <div class="doc-name">{document['filename']}</div>
                        <div class="doc-delete">✕</div>
                    </div>
                    <div class="doc-meta">
                        <span class="doc-pill">📄 {document['pages']} pages</span>
                        <span class="doc-pill">🧩 {document['chunks']} chunks</span>
                        <span class="doc-pill">● {document['status']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No documents have been indexed yet.")

    stats = get_document_stats()
    st.caption(f"Documents: {stats['document_count']} | Chunks: {stats['chunk_count']}")

if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-emoji">📚</div>
            <div class="empty-title">Ready to explore your knowledge base.</div>
            <div class="empty-copy">Upload one or more PDFs, ask questions in natural language, and receive grounded answers with citations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message.get("role", "assistant"), avatar="🤖" if message.get("role") == "assistant" else "🧑"):
        if message.get("role") == "user":
            if message.get("timestamp"):
                st.markdown(f'<div class="message-meta">{message["timestamp"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="message-shell"><div class="chat-bubble user-bubble">{message["content"]}</div></div>',
                unsafe_allow_html=True,
            )
        else:
            if message.get("timestamp"):
                st.markdown(f'<div class="message-meta">{message["timestamp"]}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="message-shell"><div class="chat-bubble assistant-bubble">{message["content"]}</div></div>',
                unsafe_allow_html=True,
            )
            if message.get("sources"):
                top_similarity = message["sources"][0].get("similarity", 0) if message.get("sources") else None
                if top_similarity is not None:
                    percentage = max(6, int(float(top_similarity) * 100))
                    color = _similarity_color(top_similarity)
                    st.markdown(
                        f'<div class="citation-card"><div class="citation-top"><div class="citation-title">Grounding confidence</div><div class="citation-meta">{int(float(top_similarity) * 100)}%</div></div><div class="similarity-bar"><div class="similarity-fill" style="width:{percentage}%; background:{color};"></div></div></div>',
                        unsafe_allow_html=True,
                    )
                for source in message["sources"]:
                    preview = _format_preview(str(source.get("content", "")))
                    similarity = source.get("similarity", 0)
                    bar_color = _similarity_color(similarity)
                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-top">
                                <div class="citation-title">📄 {source.get('source', 'unknown')}</div>
                                <div class="citation-meta">p{source.get('page', 'unknown')}</div>
                            </div>
                            <div class="citation-meta">Similarity {float(similarity) if similarity is not None else 0:.2f}</div>
                            <div class="similarity-bar"><div class="similarity-fill" style="width:{max(8, int(float(similarity) * 100))}%; background:{bar_color};"></div></div>
                            <details class="citation-preview">
                                <summary>Preview</summary>
                                <div class="citation-preview"><span class="citation-highlight">{preview}</span></div>
                            </details>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            elif message.get("is_grounded") is False:
                st.info("No relevant context was found, so the answer is not grounded in the uploaded documents.")

st.markdown('<div class="chat-input-shell">', unsafe_allow_html=True)
prompt = st.chat_input("Ask anything about your uploaded documents...", key="rag_chat_input")
st.markdown('</div>', unsafe_allow_html=True)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": datetime.now().strftime("%H:%M")})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(f'<div class="message-meta">{datetime.now().strftime("%H:%M")}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="message-shell"><div class="chat-bubble user-bubble">{prompt}</div></div>',
            unsafe_allow_html=True,
        )

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown('<div class="typing-card"><div class="typing-dots"><span></span><span></span><span></span></div><div>Searching vector database...</div></div>', unsafe_allow_html=True)
        start_time = time.perf_counter()
        with st.spinner("Retrieving relevant chunks..."):
            context_chunks = retrieve_context(prompt)
        if not context_chunks:
            answer = "I do not have enough relevant context to answer that question confidently."
            sources = []
            is_grounded = False
            st.info("No relevant context was found, so the answer is not grounded in the uploaded documents.")
        else:
            st.markdown('<div class="typing-card"><div class="typing-dots"><span></span><span></span><span></span></div><div>Generating grounded answer...</div></div>', unsafe_allow_html=True)
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
            f'<div class="message-shell"><div class="chat-bubble assistant-bubble">{answer}</div></div>',
            unsafe_allow_html=True,
        )
        if sources:
            top_similarity = sources[0].get("similarity", 0) if sources else None
            if top_similarity is not None:
                percentage = max(6, int(float(top_similarity) * 100))
                color = _similarity_color(top_similarity)
                st.markdown(
                    f'<div class="citation-card"><div class="citation-top"><div class="citation-title">Grounding confidence</div><div class="citation-meta">{int(float(top_similarity) * 100)}%</div></div><div class="similarity-bar"><div class="similarity-fill" style="width:{percentage}%; background:{color};"></div></div></div>',
                    unsafe_allow_html=True,
                )
            for source in sources:
                preview = _format_preview(str(source.get("content", "")))
                similarity = source.get("similarity", 0)
                bar_color = _similarity_color(similarity)
                st.markdown(
                    f"""
                    <div class="citation-card">
                        <div class="citation-top">
                            <div class="citation-title">📄 {source.get('source', 'unknown')}</div>
                            <div class="citation-meta">p{source.get('page', 'unknown')}</div>
                        </div>
                        <div class="citation-meta">Similarity {float(similarity) if similarity is not None else 0:.2f}</div>
                        <div class="similarity-bar"><div class="similarity-fill" style="width:{max(8, int(float(similarity) * 100))}%; background:{bar_color};"></div></div>
                        <details class="citation-preview">
                            <summary>Preview</summary>
                            <div class="citation-preview"><span class="citation-highlight">{preview}</span></div>
                        </details>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
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
