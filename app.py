"""Streamlit UI for the domain-specific RAG chatbot.

The user experience is intentionally simple: upload PDFs, ingest them into a
local vector store, ask questions in chat, and inspect the retrieved sources
used to build the response.
"""

from __future__ import annotations

import streamlit as st

from config import DATA_DIR
from generator import generate_answer
from ingest import get_document_stats, get_indexed_documents, ingest_pdfs
from retriever import retrieve_context


st.set_page_config(page_title="Domain Knowledge Assistant", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
    :root {
        --bg: #0A0E14;
        --surface: #131826;
        --surface-2: #1A2132;
        --accent: #00E5FF;
        --accent-2: #7C5CFF;
        --warning: #FFB020;
        --text: #F5F7FA;
        --muted: #96A0B6;
        --border: rgba(255,255,255,0.08);
    }
    .app-header {
        background: linear-gradient(90deg, rgba(0,229,255,0.16), rgba(124,92,255,0.09));
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.22);
    }
    .app-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: var(--text);
        margin: 0;
    }
    .app-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 0.98rem;
        color: var(--muted);
        margin-top: 0.3rem;
    }
    .stChatMessage {
        padding: 0.18rem 0 0.3rem 0;
    }
    .chat-bubble {
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin: 0.2rem 0 0.5rem 0;
        border: 1px solid var(--border);
        line-height: 1.6;
        font-family: 'Inter', sans-serif;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .user-bubble {
        background: linear-gradient(135deg, rgba(0,229,255,0.18), rgba(0,229,255,0.08));
        margin-left: 2rem;
    }
    .assistant-bubble {
        background: linear-gradient(135deg, rgba(124,92,255,0.14), rgba(19,24,38,0.95));
        margin-right: 2rem;
    }
    .metadata-text {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        font-size: 0.78rem;
        letter-spacing: 0.03em;
        color: var(--muted);
        line-height: 1.5;
    }
    .grounding-meter {
        margin: 0.4rem 0 0.6rem 0;
    }
    .grounding-bar {
        height: 6px;
        width: 100%;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
        margin-top: 0.25rem;
    }
    .grounding-fill {
        height: 100%;
        border-radius: inherit;
        transition: width 220ms ease;
    }
    .source-chip {
        display: inline-block;
        margin: 0.25rem 0.35rem 0.25rem 0;
        padding: 0.42rem 0.62rem;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(255,255,255,0.03);
        color: var(--text);
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        font-size: 0.74rem;
        letter-spacing: 0.03em;
        transition: border-color 180ms ease, color 180ms ease, transform 180ms ease;
    }
    .source-chip:hover {
        border-color: var(--accent);
        color: var(--accent);
        transform: translateY(-1px);
    }
    .stExpander {
        border: 0 !important;
        border-radius: 12px;
        background: transparent;
    }
    .stExpander summary {
        color: var(--text);
        font-weight: 600;
    }
    .sidebar .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">Domain Knowledge Assistant</div>
        <div class="app-subtitle">Ask questions, get grounded answers with citations</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

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
                st.success(
                    f"Indexed {len(result['ingested'])} document(s) and {result['chunk_count']} chunk(s)."
                )
            if result["skipped"]:
                st.info("Skipped already indexed files: " + ", ".join(result["skipped"]))

    st.divider()
    st.subheader("Indexed documents")
    documents = get_indexed_documents()
    if documents:
        for document in documents:
            st.write(f"• {document}")
    else:
        st.info("No documents have been indexed yet.")

    stats = get_document_stats()
    st.caption(f"Documents: {stats['document_count']} | Chunks: {stats['chunk_count']}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(f'<div class="chat-bubble user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble assistant-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            if message.get("sources"):
                top_similarity = message["sources"][0].get("similarity", 0) if message.get("sources") else None
                if top_similarity is not None:
                    percentage = max(4, int(float(top_similarity) * 100))
                    color = "#00E5FF" if float(top_similarity) >= 0.3 else "#FFB020"
                    st.markdown(
                        f'<div class="grounding-meter"><div class="metadata-text">Grounding: {int(float(top_similarity) * 100)}%</div><div class="grounding-bar"><div class="grounding-fill" style="width:{percentage}%; background:{color};"></div></div></div>',
                        unsafe_allow_html=True,
                    )
            if message.get("sources"):
                st.markdown('<div class="metadata-text">Sources</div>', unsafe_allow_html=True)
                for source in message["sources"]:
                    st.markdown(
                        f'<span class="source-chip">{source["source"]} · p{source["page"]}</span>',
                        unsafe_allow_html=True,
                    )
            elif message.get("is_grounded") is False:
                st.warning("No relevant context was found, so the answer is not grounded in the uploaded documents.")

prompt = st.chat_input("Ask a question about the uploaded PDFs")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(f'<div class="chat-bubble user-bubble">{prompt}</div>', unsafe_allow_html=True)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context_chunks = retrieve_context(prompt)
            if not context_chunks:
                answer = "I do not have enough relevant context to answer that question confidently."
                sources = []
                is_grounded = False
                st.warning("No relevant context was found, so the answer is not grounded in the uploaded documents.")
            else:
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

        st.markdown(f'<div class="chat-bubble assistant-bubble">{answer}</div>', unsafe_allow_html=True)
        if sources:
            top_similarity = sources[0].get("similarity", 0) if sources else None
            if top_similarity is not None:
                percentage = max(4, int(float(top_similarity) * 100))
                color = "#00E5FF" if float(top_similarity) >= 0.3 else "#FFB020"
                st.markdown(
                    f'<div class="grounding-meter"><div class="metadata-text">Grounding: {int(float(top_similarity) * 100)}%</div><div class="grounding-bar"><div class="grounding-fill" style="width:{percentage}%; background:{color};"></div></div></div>',
                    unsafe_allow_html=True,
                )
            st.markdown('<div class="metadata-text">Sources</div>', unsafe_allow_html=True)
            for source in sources:
                st.markdown(
                    f'<span class="source-chip">{source["source"]} · p{source["page"]}</span>',
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
        }
    )
    st.rerun()
