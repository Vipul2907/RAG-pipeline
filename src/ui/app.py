#!/usr/bin/env python3
"""
RAG Pipeline Web UI - Streamlit Dashboard

A browser-based interface for:
- File upload and dynamic ingestion into HNSW index
- Diagnostic view with index statistics
- Chat interface with grounded answers and citations

Run with: streamlit run src/ui/app.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import config, setup_logging
from rag.ingestion import DocumentChunker
from rag.embeddings import get_embedding_provider
from rag.index import BruteForceIndex, HNSWIndex
from rag.query import QueryEngine, RAGResult, Citation
from rag.llm import get_llm_provider


# Page configuration
st.set_page_config(
    page_title="RAG Pipeline - HNSW Vector Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .citation-box {
        background: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .citation-source {
        font-weight: 600;
        color: #1f77b4;
    }
    .citation-score {
        color: #888;
        font-size: 0.8rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background: #f5f5f5;
        margin-right: 2rem;
    }
    .stats-table {
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "index": None,
        "index_type": "hnsw",
        "engine": None,
        "chunks_count": 0,
        "embedding_provider": None,
        "chat_history": [],
        "last_query_time": 0,
        "index_stats": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_cached_embedding_provider(provider_type: str, **kwargs):
    """Cached embedding provider factory."""
    from rag.embeddings import get_embedding_provider as _get_provider
    return _get_provider(provider_type, **kwargs)


def build_index_from_files(uploaded_files, index_type, chunk_size, chunk_overlap, embedding_provider):
    """Build index from uploaded files."""
    from rag.ingestion import Chunk

    all_chunks = []
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=chunk_overlap)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 0.5) / len(uploaded_files))

        # Read file content
        content = uploaded_file.read().decode("utf-8")
        chunks = chunker.chunk_text(content, uploaded_file.name)
        all_chunks.extend(chunks)

    progress_bar.progress(0.5)
    status_text.text("Generating embeddings...")

    # Generate embeddings
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedding_provider.embed(texts)

    progress_bar.progress(0.7)
    status_text.text(f"Building {index_type.upper()} index...")

    # Create index
    if index_type == "hnsw":
        index = HNSWIndex(
            dimension=embedding_provider.dimension,
            M=16,
            ef_construction=200,
            ef_search=50,
        )
    else:
        index = BruteForceIndex(dimension=embedding_provider.dimension)

    index.add(embeddings, all_chunks)

    progress_bar.progress(1.0)
    status_text.text("Done!")

    return index, all_chunks


def render_sidebar():
    """Render sidebar with configuration and file upload."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        # Index type selection
        index_type = st.radio(
            "Index Type",
            ["hnsw", "brute_force"],
            index=0 if st.session_state.index_type == "hnsw" else 1,
            help="HNSW for approximate search (faster at scale), BruteForce for exact search"
        )
        st.session_state.index_type = index_type

        st.markdown("---")

        # Chunking parameters
        st.markdown("### 📄 Chunking")
        chunk_size = st.slider("Chunk Size", 128, 2048, 512, 64)
        chunk_overlap = st.slider("Chunk Overlap", 0, 256, 50, 16)

        st.markdown("---")

        # Embedding provider
        st.markdown("### 🧠 Embeddings")
        embedding_provider_type = st.selectbox(
            "Provider",
            ["dummy", "huggingface", "openai"],
            index=0,
            help="Dummy: deterministic hashing (no deps). HuggingFace: local sentence-transformers. OpenAI: API-based."
        )

        if embedding_provider_type == "huggingface":
            model_name = st.selectbox(
                "Model",
                [
                    "sentence-transformers/all-MiniLM-L6-v2",
                    "sentence-transformers/all-MiniLM-L12-v2",
                    "sentence-transformers/all-mpnet-base-v2",
                ],
                index=0,
            )
        elif embedding_provider_type == "openai":
            model_name = st.selectbox(
                "Model",
                ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
                index=0,
            )
        else:
            model_name = None

        # LLM Provider
        st.markdown("---")
        st.markdown("### 🤖 LLM Provider")
        llm_provider = st.selectbox(
            "Provider",
            ["anthropic", "openai", "groq"],
            index=0,
        )
        if llm_provider == "anthropic":
            llm_model = st.selectbox("Model", ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"], index=0)
        elif llm_provider == "openai":
            llm_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
        else:  # groq
            llm_model = st.selectbox("Model", ["openai/gpt-oss-120b", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"], index=0)

        st.markdown("---")

        # File upload
        st.markdown("### 📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drop .txt or .md files",
            type=["txt", "md"],
            accept_multiple_files=True,
            help="Upload documents to ingest into the vector index"
        )

        if uploaded_files:
            if st.button("🚀 Ingest & Build Index", type="primary", use_container_width=True):
                with st.spinner("Building index..."):
                    try:
                        # Get embedding provider
                        if embedding_provider_type == "dummy":
                            emb_provider = get_embedding_provider("dummy")
                        elif embedding_provider_type == "huggingface":
                            emb_provider = get_embedding_provider("huggingface", model_name=model_name)
                        else:
                            emb_provider = get_embedding_provider("openai", model_name=model_name)

                        # Build index
                        index, chunks = build_index_from_files(
                            uploaded_files, index_type, chunk_size, chunk_overlap, emb_provider
                        )

                        # Get LLM provider
                        llm = get_llm_provider(llm_provider, model_name=llm_model)

                        # Create query engine
                        engine = QueryEngine(index, emb_provider, llm)

                        # Save to session
                        st.session_state.index = index
                        st.session_state.engine = engine
                        st.session_state.embedding_provider = emb_provider
                        st.session_state.chunks_count = len(chunks)
                        st.session_state.index_stats = index.get_stats() if hasattr(index, 'get_stats') else {
                            "size": index.size,
                            "dimension": index.dimension,
                        }

                        st.success(f"✅ Indexed {len(chunks)} chunks from {len(uploaded_files)} files!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error: {e}")

        # Index info
        if st.session_state.index:
            st.markdown("---")
            st.markdown("### 📊 Current Index")
            st.info(f"""
            **Type:** {st.session_state.index_type.upper()}
            **Vectors:** {st.session_state.chunks_count}
            **Dimension:** {st.session_state.index.dimension}
            """)


def render_diagnostics():
    """Render diagnostic view with index statistics."""
    st.markdown("## 📊 Diagnostics")

    if not st.session_state.index:
        st.warning("No index built yet. Upload files and click 'Ingest & Build Index' in the sidebar.")
        return

    index = st.session_state.index

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📦 Vectors", st.session_state.chunks_count)

    with col2:
        st.metric("📐 Dimension", index.dimension)

    with col3:
        if hasattr(index, 'get_stats'):
            stats = index.get_stats()
            st.metric("🏗️ Max Layer", stats.get('max_layer', 'N/A'))
        else:
            st.metric("🏗️ Max Layer", "N/A (BruteForce)")

    with col4:
        st.metric("🔧 Index Type", st.session_state.index_type.upper())

    # Detailed stats table
    if hasattr(index, 'get_stats'):
        stats = index.get_stats()

        st.markdown("### 📈 HNSW Structure")
        stats_data = {
            "Metric": [
                "Total Vectors",
                "Vector Dimension",
                "Number of Layers",
                "Max Layer",
                "Entry Point Node",
                "M (max connections)",
                "M0 (layer 0 connections)",
                "ef_construction",
                "ef_search",
                "Avg Links/Node",
            ],
            "Value": [
                stats.get('size', 'N/A'),
                stats.get('dimension', 'N/A'),
                stats.get('num_layers', 'N/A'),
                stats.get('max_layer', 'N/A'),
                stats.get('entry_point', 'N/A'),
                stats.get('M', 'N/A'),
                stats.get('M0', 'N/A'),
                stats.get('ef_construction', 'N/A'),
                stats.get('ef_search', 'N/A'),
                f"{stats.get('avg_links_per_node', 0):.1f}",
            ],
        }
        st.table(stats_data)

        # Layer distribution
        if 'nodes_per_layer' in stats:
            st.markdown("### 📊 Nodes per Layer")
            layer_data = {
                "Layer": list(range(len(stats['nodes_per_layer']))),
                "Nodes": stats['nodes_per_layer'],
            }
            st.bar_chart(layer_data, x="Layer", y="Nodes")


def render_chat():
    """Render chat interface."""
    st.markdown("## 💬 Chat with Your Documents")

    if not st.session_state.engine:
        st.info("Build an index first to start chatting!")
        return

    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message"><b>You:</b> {message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message"><b>Assistant:</b> {message["content"]}</div>',
                unsafe_allow_html=True
            )
            if message.get("citations"):
                with st.expander("📚 Citations", expanded=False):
                    for cite in message["citations"]:
                        st.markdown(f"""
                        <div class="citation-box">
                            <span class="citation-source">Source: {cite['source_file']}, chunk {cite['chunk_index']}</span>
                            <span class="citation-score"> (score: {cite['score']:.3f})</span>
                            <br><br>
                            {cite['text_snippet']}
                        </div>
                        """, unsafe_allow_html=True)

    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])
        with col1:
            query = st.text_input(
                "Ask a question about your documents:",
                placeholder="e.g., What is machine learning?",
                label_visibility="collapsed"
            )
        with col2:
            submitted = st.form_submit_button("Send", type="primary", use_container_width=True)

    if submitted and query:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": query})

        # Get RAG result
        with st.spinner("Thinking..."):
            start_time = time.time()
            try:
                result: RAGResult = st.session_state.engine.generate(query, k=10)
                elapsed = time.time() - start_time
            except Exception as e:
                st.error(f"Generation error: {e}")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "citations": []
                })
                st.rerun()

        # Add assistant message
        citations_data = [
            {
                "source_file": c.source_file,
                "chunk_index": c.chunk_index,
                "text_snippet": c.text_snippet,
                "score": c.score,
            }
            for c in result.citations
        ]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.answer,
            "citations": citations_data,
            "time": elapsed,
        })

        st.session_state.last_query_time = elapsed
        st.rerun()

    # Show last query time
    if st.session_state.last_query_time > 0:
        st.caption(f"⏱️ Last query: {st.session_state.last_query_time:.2f}s")


def render_retrieval_debug():
    """Render retrieval debug panel."""
    if not st.session_state.engine:
        return

    with st.expander("🔍 Retrieval Debug", expanded=False):
        st.markdown("Test retrieval without LLM generation")

        debug_query = st.text_input("Debug query:", placeholder="Enter query to test retrieval...")
        debug_k = st.slider("Top K", 1, 20, 10)

        if st.button("Search", key="debug_search") and debug_query:
            with st.spinner("Searching..."):
                result = st.session_state.engine.search(debug_query, k=debug_k)

            st.markdown(f"**Found {len(result.results)} results:**")

            for i, r in enumerate(result.results):
                with st.container():
                    st.markdown(f"""
                    **Rank {r.rank}** | Score: {r.score:.4f} | Source: `{r.chunk.source_file}` (chunk {r.chunk.chunk_index})
                    """)
                    st.text(r.chunk.text[:500] + ("..." if len(r.chunk.text) > 500 else ""))


def main():
    """Main Streamlit app."""
    init_session_state()
    setup_logging(config)

    # Header
    st.markdown('<div class="main-header">🔍 RAG Pipeline Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Custom HNSW Vector Index + LLM Grounded Generation</div>', unsafe_allow_html=True)

    # Sidebar
    render_sidebar()

    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Diagnostics", "🔍 Retrieval Debug"])

    with tab1:
        render_chat()

    with tab2:
        render_diagnostics()

    with tab3:
        render_retrieval_debug()

    # Footer
    st.markdown("---")
    st.caption("RAG Pipeline • Hand-rolled HNSW Index • Built with Streamlit")


if __name__ == "__main__":
    main()