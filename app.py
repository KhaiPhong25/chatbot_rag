# =============================================================================
# STREAMLIT WEB UI - Giao diện hỏi đáp trực quan
# Chạy bằng: streamlit run app.py
# Tích hợp: Streaming, Hybrid Search, Incremental Indexing, Multi-Model Fallback
# =============================================================================

import os
import streamlit as st

from rag_core import (
    get_embedding_model,
    build_or_update_vectorstore,
    build_hybrid_retriever,
    stream_with_fallback,
    FREE_MODELS,
    CHAT_HISTORY_FILE,
)

# ─────────────────────────── Cấu hình trang ───────────────────────────
st.set_page_config(
    page_title="Chatbot Lịch Sử Đảng CSVN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── CSS tùy chỉnh ────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .app-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .app-header h1 {
        color: #fff;
        margin: 0;
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: rgba(255,255,255,0.65);
        margin: 6px 0 0;
        font-size: 0.92rem;
    }

    /* Badges */
    .badge-row {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin-top: 12px;
    }
    .badge {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        color: #e2e8f0;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
    }

    /* Model label */
    .model-label {
        font-size: 0.72rem;
        color: #94a3b8;
        margin-bottom: 4px;
        letter-spacing: 0.04em;
    }

    /* Chat input */
    .stChatInput > div {
        border-radius: 12px !important;
        border: 1.5px solid #334155 !important;
        background: #1e293b !important;
    }

    /* Source pills */
    .source-pill {
        display: inline-block;
        background: rgba(99,102,241,0.15);
        border: 1px solid rgba(99,102,241,0.3);
        color: #a5b4fc;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.76rem;
        margin: 2px;
    }

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {
        background: #0f172a;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────── Khởi tạo RAG (cache theo session) ────────────────
@st.cache_resource(show_spinner="⚙️ Đang khởi động hệ thống RAG... (chỉ chạy một lần)")
def init_rag():
    """Singleton: khởi tạo embedding model, vectorstore, hybrid retriever một lần."""
    emb        = get_embedding_model()
    vs         = build_or_update_vectorstore(emb)
    hybrid_ret = build_hybrid_retriever(vs, k=5)
    return hybrid_ret


retriever = init_rag()


# ─────────────────────────── Session State ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role": "user"|"assistant", "content": str, "model": str|None}

if "session_id" not in st.session_state:
    st.session_state.session_id = "web_user"


# ─────────────────────────── SIDEBAR ─────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Cấu hình")

    st.markdown("**🤖 Thứ tự ưu tiên mô hình:**")
    for i, m in enumerate(FREE_MODELS, 1):
        short = m.split("/")[-1].replace(":free", "")
        label = "🟢 Ưu tiên" if i == 1 else f"🔵 Fallback {i-1}"
        st.markdown(f"`{label}` **{short}**")

    st.divider()

    st.markdown("**📊 Cấu hình Hybrid Search:**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("BM25", "40%", help="Tìm kiếm từ khóa chính xác")
    with col2:
        st.metric("FAISS", "60%", help="Tìm kiếm ngữ nghĩa vector")

    st.divider()

    if st.button("🗑️ Xóa lịch sử hội thoại", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        if os.path.exists(CHAT_HISTORY_FILE):
            os.remove(CHAT_HISTORY_FILE)
        st.success("Đã xóa lịch sử! Bắt đầu cuộc hội thoại mới.")

    st.divider()
    st.caption(
        "**RAG Chatbot v2.0**\n\n"
        "Semantic Chunking · Hybrid Search · Streaming · Multi-Model Fallback"
    )


# ─────────────────────────── HEADER ──────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🎓 Chatbot Hỏi Đáp Lịch Sử Đảng CSVN</h1>
    <p>Hỏi đáp thông minh từ giáo trình, trích dẫn số trang, trí nhớ đa lượt</p>
    <div class="badge-row">
        <span class="badge">✂️ SemanticChunker</span>
        <span class="badge">🔍 Hybrid Search BM25+FAISS</span>
        <span class="badge">⚡ Streaming</span>
        <span class="badge">🔄 Multi-Model Fallback</span>
        <span class="badge">📄 Incremental Indexing</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────── Hiển thị lịch sử chat ───────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        if msg["role"] == "assistant" and msg.get("model"):
            st.markdown(
                f'<div class="model-label">via {msg["model"]}</div>',
                unsafe_allow_html=True
            )
        st.markdown(msg["content"])


# ─────────────────────────── Chat Input ──────────────────────────────
prompt = st.chat_input("Nhập câu hỏi về Lịch Sử Đảng Cộng Sản Việt Nam...")

if prompt:
    # Hiển thị tin nhắn người dùng
    st.session_state.messages.append({"role": "user", "content": prompt, "model": None})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Tạo phản hồi với streaming
    with st.chat_message("assistant", avatar="🤖"):
        model_placeholder = st.empty()
        response_box      = st.empty()

        full_response = ""
        used_model    = None

        try:
            for token, model_name in stream_with_fallback(
                prompt,
                retriever,
                session_id=st.session_state.session_id,
            ):
                if used_model is None:
                    used_model = model_name
                    model_placeholder.markdown(
                        f'<div class="model-label">via {model_name}</div>',
                        unsafe_allow_html=True
                    )

                full_response += token
                # Hiện chữ đang gõ (cursor ▌)
                response_box.markdown(full_response + " ▌")

            # Xóa cursor khi hoàn tất
            response_box.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Lỗi khi kết nối mô hình: {e}"
            response_box.error(full_response)

        # Lưu vào session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "model": used_model,
        })
